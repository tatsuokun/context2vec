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

        self.W = nn.Embedding(num_embeddings=len(counter),
                              embedding_dim=embed_size,
                              padding_idx=ignore_index)

        def forward(self, sentence):
            quit()
