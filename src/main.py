import numpy as np
import os
import torch
from torch import optim
from src.util.args import parse_args
from src.util.batch import Dataset
from src.util.io import write_embedding, write_config, read_config, load_vocab
from src.nets import Context2vec


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
    model.norm_embedding_weight(model.criterion.W)
    '''

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
        topv, topi = model.run_inference(input_tokens, target_pos)
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

    if train:
        batch_size = args.batch_size
        n_epochs = args.epoch
        word_embed_size = args.hidden_size
        hidden_size = args.hidden_size
        learning_rate = 1e-3
        if not os.path.isfile(args.input_file):
            raise FileNotFoundError

        with open(args.input_file) as f:
            sentences = [line.strip().lower().split() for line in f]

        dataset = Dataset(sentences, batch_size, args.freq_min, device)
        counter = np.array([dataset.vocab.freqs[word] if word in dataset.vocab.freqs else 0
                            for word in dataset.vocab.itos])
        model = Context2vec(vocab_size=len(dataset.vocab),
                            counter=counter,
                            word_embed_size=word_embed_size,
                            hidden_size=hidden_size,
                            n_layers=1,
                            bidirectional=True,
                            use_mlp=True,
                            dropout=0.0,
                            pad_index=dataset.pad_index,
                            device=device,
                            inference=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(batch_size, n_epochs, word_embed_size, hidden_size, device)
        print(model)

        for epoch in range(args.epoch):
            total_loss = 0.0
            for iterator in dataset.get_batch_iter(batch_size):
                for batch in iterator:
                    sentence = getattr(batch, 'sentence')
                    target = sentence[:, 1:-1]
                    if target.size(1) == 0:
                        continue
                    optimizer.zero_grad()
                    loss = model(sentence, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data.mean()
            print(total_loss.item())

        output_dir = os.path.dirname(args.wordsfile)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        write_embedding(dataset.vocab.itos, model.criterion.W, use_cuda, args.wordsfile)
        torch.save(model.state_dict(), args.modelfile)
        torch.save(optimizer.state_dict(), args.modelfile+'.optim')
        config_file = args.modelfile+'.config.json'
        write_config(config_file,
                     vocab_size=len(dataset.vocab),
                     word_embed_size=word_embed_size,
                     hidden_size=hidden_size,
                     n_layers=1,
                     bidirectional=True,
                     use_mlp=True,
                     dropout=0.0,
                     pad_index=dataset.pad_index,
                     unk_token=dataset.unk_token,
                     bos_token=dataset.bos_token,
                     eos_token=dataset.eos_token,
                     learning_rate=learning_rate)
    else:
        config_file = args.modelfile+'.config.json'
        config_dict = read_config(config_file)
        model = Context2vec(vocab_size=config_dict['vocab_size'],
                            counter=[0]*config_dict['vocab_size'],
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

        run_inference_by_user_input(model,
                                    itos,
                                    stoi,
                                    unk_token=unk_token,
                                    bos_token=bos_token,
                                    eos_token=eos_token,
                                    device=device)
