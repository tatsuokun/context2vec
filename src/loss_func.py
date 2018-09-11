import torch
import torch.nn as nn


class NegativeSampling(nn.Module):

    def __init__(self,
                 embed_size,
                 counter,
                 n_negatives,
                 power,
                 ignore_index):
        super(NegativeSampling, self).__init__()

        self.W = nn.Embedding(num_embeddings=len(counter)+4,
                              embedding_dim=embed_size,
                              padding_idx=ignore_index)
        self.W.weight.data.zero_()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, sentence, context):
        batch_size, _ = sentence.size()
        emb = self.W(sentence)
        pos = self.logsigmoid(torch.mul(emb, context).sum(2)).sum()
        return -pos/batch_size
