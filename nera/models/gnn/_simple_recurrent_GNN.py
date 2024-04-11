import copy
from typing import Optional

import torch
import torch.nn as nn

from ._GConvRNN import GConvElman
from ._gconv_gru_copy import GConvGRU


#from torch_geometric_temporal.nn import GConvGRU


class RGNN(nn.Module):
    _activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'lrelu': nn.LeakyReLU(0.2),
        'tanh': nn.Tanh(),
    }

    def __init__(self, team_count: int,
                 embed_dim: int = 20,
                 dense_dims: tuple[int] = (8, 4),
                 conv_out_channels: int = 1,
                 conv_hidden_channels: int = 1,
                 dropout_rate: float = 0.1,
                 activation: str = 'relu',
                 target_dim: int = 3,
                 K: int = 1,
                 bias: bool = True,
                 init_ones_: bool = False,
                 graph_conv: str = 'GCONV_GRU',
                 normalization: Optional[str] = 'sym',
                 debug=False):

        assert K > 0
        assert graph_conv.upper() in ['GCONV_GRU', 'GCONV_ELMAN']

        super(RGNN, self).__init__()

        self.team_count = team_count
        self.act_fn = self._activations[activation]

        self.embed_dim = embed_dim
        self.embedding = nn.Parameter(torch.full((team_count, embed_dim), 1, dtype=torch.float))
        #self.embedding = nn.Embedding(num_embeddings=team_count, embedding_dim=embed_dim)

        # recurrent graph convolution layer
        if graph_conv.upper() == 'GCONV_ELMAN':
            gconv = GConvElman(embed_dim, conv_out_channels,
                               aggr="add", init_ones_=init_ones_, bias=bias, hidden_channels=conv_hidden_channels)
        else:
            # graph_conv == 'GCONV_GRU'
            gconv = GConvGRU(embed_dim, conv_out_channels, K, bias=bias,
                             init_ones_=init_ones_, normalization=normalization, )
        self.rnn_gconv = gconv

        self.debug = debug
        self.embedding_progression = []
        self.H_progression = []

        # simple fully connected part for prediction
        dense = nn.Sequential()
        dense.add_module('lin0', nn.Linear(in_features=conv_out_channels * 2, out_features=dense_dims[0]))
        #dense.add_module('act0', self.act_fn)
        #dense.add_module('dropout0', nn.Dropout(p=dropout_rate))
        for i in range(1, len(dense_dims)):
            dense.add_module(f'lin{i}', nn.Linear(in_features=dense_dims[i - 1], out_features=dense_dims[i]))
            #dense.add_module(f'act{i}', self.act_fn)
            #dense.add_module(f'dropout{i}', nn.Dropout(p=dropout_rate))
        dense.add_module(f'lin{len(dense_dims)}',
                         nn.Linear(in_features=dense_dims[len(dense_dims) - 1], out_features=target_dim))
        #dense.add_module(f'act{len(dense_dims)}', self.act_fn)
        #dense.add_module(f'dropout{len(dense_dims)}', nn.Dropout(p=dropout_rate))
        self.dense_sequence = dense

        if init_ones_:
            for module in self.dense_sequence.children():
                if isinstance(module, nn.Linear):
                    nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        self.out_layer = nn.Softmax(dim=0)

    def forward(self, edge_index, home, away, edge_weight=None):
        h = torch.tensor(list(range(self.team_count)))
        h = self.embedding[h].reshape(-1, self.embed_dim)
        if self.debug:
            self.embedding_progression.append(h.detach().clone().numpy())

        # graph convolution
        h = self.rnn_gconv(h, edge_index, edge_weight)

        # prediction
        h = torch.cat([h[home], h[away]], dim=0)
        #h = self.dense_sequence(h)
        h = self.out_layer(h)
        return h
