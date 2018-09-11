import torch
import torch.nn as nn
import numpy as np


class NegativeSampling(nn.Module):

    def __init__(self,
                 embed_size,
                 counter,
                 n_negatives,
                 power,
                 device,
                 ignore_index):
        super(NegativeSampling, self).__init__()

        self.counter = counter
        self.n_negatives = n_negatives
        self.power = power
        self.device = device

        self.W = nn.Embedding(num_embeddings=len(counter),
                              embedding_dim=embed_size,
                              padding_idx=ignore_index)
        self.W.weight.data.zero_()
        self.logsigmoid = nn.LogSigmoid()
        self.negative_table = self.init_negative_table(counter)

    def init_negative_table(self, counter, table_length=10*7):
        self.negative_table_size = table_length
        z = np.sum(np.power(counter, self.power))
        negative_table = np.zeros(table_length, dtype=np.int32)
        begin_index = 0
        for word_id, freq in enumerate(counter):
            c = np.power(freq, self.power)
            end_index = begin_index + int(c * table_length / z) + 1
            negative_table[begin_index:end_index] = word_id
            begin_index = end_index
        return negative_table

    def negative_sampling(self, num_negatives, size):
        if num_negatives > 0:
            negatives = self.negative_table[np.random.randint(low=0,
                                                              high=self.negative_table_size,
                                                              size=size)]
            negatives = torch.from_numpy(negatives).long().to(self.device)
            return negatives
        else:
            raise NotImplementedError

    def forward(self, sentence, context):
        batch_size, seq_len = sentence.size()
        emb = self.W(sentence)
        pos_loss = self.logsigmoid((emb*context).sum(2)).sum()

        neg_samples = self.negative_sampling(self.n_negatives, (batch_size, seq_len, self.n_negatives))
        neg_emb = self.W(neg_samples)
        neg_loss = self.logsigmoid((-neg_emb*context.unsqueeze(2)).sum(3)).sum()
        return -(pos_loss+neg_loss)/batch_size
