import numpy as np
import os
import time
import torch
from torch import optim
from src.eval.mscc import mscc_evaluation
from src.core.nets import Context2vec
from src.util.args import parse_args
from src.util.batch import Dataset
from src.util.config import Config
from src.util.io import write_embedding, write_config, read_config, load_vocab


def run_inference_by_user_input(model,
                                itos,
                                stoi,
                                unk_token,
                                bos_token,
                                eos_token,
                                device):

    def return_split_sentence(sentence):
        if ' ' not in sentence:
            print('sentence should contain white space to split it into tokens')
            raise SyntaxError
        elif '[]' not in sentence:
            print('sentence should contain `[]` that notes the target')
            raise SyntaxError
        else:
            tokens = sentence.lower().strip().split()
            target_pos = tokens.index('[]')
            return tokens, target_pos

    ''' norm_weight
    '''
    model.norm_embedding_weight(model.criterion.W)

    while True:
        sentence = input('>> ')
        try:
            tokens, target_pos = return_split_sentence(sentence)
        except SyntaxError:
            continue
        tokens[target_pos] = unk_token
        tokens = [bos_token] + tokens + [eos_token]
        indexed_sentence = [stoi[token] if token in stoi else stoi[unk_token] for token in tokens]
        input_tokens = \
            torch.tensor(indexed_sentence, dtype=torch.long, device=device).unsqueeze(0)
        topv, topi = model.run_inference(input_tokens, target=None, target_pos=target_pos)
        for value, key in zip(topv, topi):
            print(value.item(), itos[key.item()])


def main():
    args = parse_args()
    gpu_id = args.gpu_id
    train = args.train
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    if use_cuda:
        device = torch.device('cuda:{}'.format(gpu_id))
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    config = Config(args.config_file)

    if train:
        batch_size = config.batch_size
        n_epochs = config.n_epochs
        word_embed_size = config.word_embed_size
        hidden_size = config.hidden_size
        learning_rate = config.learning_rate
        if not os.path.isfile(args.input_file):
            raise FileNotFoundError

        with open(args.input_file) as f:
            sentences = [line.strip().lower().split() for line in f]

        dataset = Dataset(sentences, batch_size, config.min_freq, device)
        counter = np.array([dataset.vocab.freqs[word] if word in dataset.vocab.freqs else 0
                            for word in dataset.vocab.itos])
        model = Context2vec(vocab_size=len(dataset.vocab),
                            counter=counter,
                            word_embed_size=word_embed_size,
                            hidden_size=hidden_size,
                            n_layers=config.n_layers,
                            bidirectional=True,
                            use_mlp=config.use_mlp,
                            dropout=config.dropout,
                            pad_index=dataset.pad_index,
                            device=device,
                            inference=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(batch_size, n_epochs, word_embed_size, hidden_size, device)
        print(model)

        interval = 1e6
        for epoch in range(n_epochs):
            begin_time = time.time()
            cur_at = begin_time
            total_loss = 0.0
            word_count = 0
            next_count = interval
            last_accum_loss = 0.0
            last_word_count = 0
            for iterator in dataset.get_batch_iter(batch_size):
                for batch in iterator:
                    sentence = getattr(batch, 'sentence')
                    target = sentence[:, 1:-1]
                    if target.size(0) == 0:
                        continue
                    optimizer.zero_grad()
                    loss = model(sentence, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data.mean()

                    minibatch_size, sentence_length = target.size()
                    word_count += minibatch_size * sentence_length
                    accum_mean_loss = float(total_loss)/word_count if total_loss > 0.0 else 0.0
                    if word_count >= next_count:
                        now = time.time()
                        duration = now - cur_at
                        throuput = float((word_count-last_word_count)) / (now - cur_at)
                        cur_mean_loss = (float(total_loss)-last_accum_loss)/(word_count-last_word_count)
                        print('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'
                              .format(word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
                        next_count += interval
                        cur_at = now
                        last_accum_loss = float(total_loss)
                        last_word_count = word_count

            print(total_loss.item())

        output_dir = os.path.dirname(args.wordsfile)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        write_embedding(dataset.vocab.itos, model.criterion.W, use_cuda, args.wordsfile)
        torch.save(model.state_dict(), args.modelfile)
        torch.save(optimizer.state_dict(), args.modelfile+'.optim')
        output_config_file = args.modelfile+'.config.json'
        write_config(output_config_file,
                     vocab_size=len(dataset.vocab),
                     word_embed_size=word_embed_size,
                     hidden_size=hidden_size,
                     n_layers=config.n_layers,
                     bidirectional=True,
                     use_mlp=config.use_mlp,
                     dropout=config.dropout,
                     pad_index=dataset.pad_index,
                     unk_token=dataset.unk_token,
                     bos_token=dataset.bos_token,
                     eos_token=dataset.eos_token,
                     learning_rate=learning_rate)
    else:
        config_file = args.modelfile+'.config.json'
        config_dict = read_config(config_file)
        model = Context2vec(vocab_size=config_dict['vocab_size'],
                            counter=[1]*config_dict['vocab_size'],
                            word_embed_size=config_dict['word_embed_size'],
                            hidden_size=config_dict['hidden_size'],
                            n_layers=config_dict['n_layers'],
                            bidirectional=config_dict['bidirectional'],
                            use_mlp=config_dict['use_mlp'],
                            dropout=config_dict['dropout'],
                            pad_index=config_dict['pad_index'],
                            device=device,
                            inference=True).to(device)
        model.load_state_dict(torch.load(args.modelfile))
        optimizer = optim.Adam(model.parameters(), lr=config_dict['learning_rate'])
        optimizer.load_state_dict(torch.load(args.modelfile+'.optim'))
        itos, stoi = load_vocab(args.wordsfile)
        unk_token = config_dict['unk_token']
        bos_token = config_dict['bos_token']
        eos_token = config_dict['eos_token']
        model.eval()

        if args.task == 'mscc':
            if not os.path.isfile(config.question_file) or not os.path.isfile(config.answer_file):
                raise FileNotFoundError

            mscc_evaluation(config.question_file,
                            config.answer_file,
                            'mscc.result',
                            model,
                            stoi,
                            unk_token=unk_token,
                            bos_token=bos_token,
                            eos_token=eos_token,
                            device=device)

        else:
            run_inference_by_user_input(model,
                                        itos,
                                        stoi,
                                        unk_token=unk_token,
                                        bos_token=bos_token,
                                        eos_token=eos_token,
                                        device=device)
