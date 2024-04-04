import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU


class RGNN(nn.Module):
    _activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(0.2)
    }

    def __init__(self, team_count: int,
                 embed_dim: int = 20,
                 dense_dims: tuple[int] = (8, 8, 8, 8, 8),
                 conv_out_channels: int = 16,
                 dropout_rate: float = 0.1,
                 activation: str = 'relu',
                 target_dim: int = 3):

        super(RGNN, self).__init__()

        self.team_count = team_count
        self.act_fn = self._activations[activation]

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=team_count, embedding_dim=embed_dim)

        # recurrent graph convolution layer
        gconv = GConvGRU(embed_dim, conv_out_channels, 1)
        self.rnn_gconv = gconv

        # simple fully connected part for prediction
        dense = nn.Sequential()
        dense.add_module('lin0', nn.Linear(in_features=conv_out_channels * 2, out_features=dense_dims[0]))
        dense.add_module('act0', self.act_fn)
        dense.add_module('dropout0', nn.Dropout(p=dropout_rate))
        for i in range(1, len(dense_dims)):
            dense.add_module(f'lin{i}', nn.Linear(in_features=dense_dims[i - 1], out_features=dense_dims[i]))
            dense.add_module(f'act{i}', self.act_fn)
            dense.add_module(f'dropout{i}', nn.Dropout(p=dropout_rate))
        dense.add_module(f'lin{len(dense_dims)}',
                         nn.Linear(in_features=dense_dims[len(dense_dims) - 1], out_features=target_dim))
        dense.add_module(f'act{len(dense_dims)}', self.act_fn)
        dense.add_module(f'dropout{len(dense_dims)}', nn.Dropout(p=dropout_rate))
        self.dense_sequence = dense

        self.out_layer = nn.Softmax(dim=1)

    def forward(self, edge_index):
        home, away = edge_index

        h = torch.tensor(list(range(self.team_count)))
        h = self.embedding(h).reshape(-1, self.embed_dim)

        # graph convolution
        h = self.rnn_gconv(h, edge_index)
        h = self.act_fn(h)

        # prediction
        h = torch.cat([h[home], h[away]], dim=1)
        h = self.dense_sequence(h)
        h = self.out_layer(h)
        return h
