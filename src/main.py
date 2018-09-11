import os
import torch
from torch import optim
import numpy as np
from src.util.args import parse_args
from src.util.batch import Dataset
from src.util.io import write_embedding
from src.nets import Context2vec


def main():
    corpus_filename = 'dataset/sample.txt'

    args = parse_args()
    gpu_id = args.gpu_id
    batch_size = args.batch_size
    n_epochs = args.epoch
    word_embed_size = args.hidden_size
    hidden_size = args.hidden_size
    use_cuda = torch.cuda.is_available() and gpu_id > -1
    if use_cuda:
        device = torch.device('cuda:{}'.format(gpu_id))
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')

    if not os.path.isfile(corpus_filename):
        raise FileNotFoundError
    with open(corpus_filename) as f:
        sentences = [line.strip().lower().split() for line in f]

    dataset = Dataset(sentences, batch_size, device)
    counter = np.array([dataset.vocab.freqs[word] if word in dataset.vocab.freqs else 0
                        for word in dataset.vocab.itos])
    model = Context2vec(vocab_size=len(dataset.vocab),
                        counter=counter,
                        word_embed_size=word_embed_size,
                        hidden_size=hidden_size,
                        n_layers=1,
                        bidirectional=True,
                        dropout=0.0,
                        pad_index=dataset.pad_index,
                        device=device,
                        inference=False)
    if use_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(batch_size, n_epochs, word_embed_size, hidden_size, device)
    print(model)

    for epoch in range(args.epoch):
        total_loss = 0.0
        for iterator in dataset.get_batch_iter(batch_size):
            for batch in iterator:
                sentence = getattr(batch, 'sentence')
                loss = model(sentence)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.data.mean()
        print(total_loss.item())
    write_embedding(dataset.vocab.itos, model.criterion.W, use_cuda, 'test_embedding.vec')
