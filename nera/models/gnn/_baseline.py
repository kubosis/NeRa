import copy
from typing import Optional

import torch
import torch.nn as nn

from ._GConvRNN import GConvElman
from ._gconv_gru_copy import GConvGRU
from ._reccurent_gnn import RecurrentGNN


#from torch_geometric_temporal.nn import GConvGRU


class RGNN(RecurrentGNN):
    _activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'lrelu': nn.LeakyReLU(0.2),
        'tanh': nn.Tanh(),
    }

    def __init__(self, team_count: int,
                 in_channels: int,
                 out_channels: int,
                 conv_out_channels: int = 1,
                 conv_hidden_channels: int = 1,
                 activation: str = 'relu',
                 K: int = 1,
                 bias: bool = True,
                 init_ones_: bool = False,
                 graph_conv: str = 'GCONV_GRU',
                 normalization: Optional[str] = 'sym',
                 discount: float = 0.5,
                 aggr: str = 'add',
                 debug=False):

        """
        Baseline recurrent graph neural network for match outcome prediction. The architecture is following:
        signal x (match home vs away team)
        h = embedding(teams)
        recurrent graph convolutional layer
        concat embeddings[away], embeddings[home]
        linear layer
        softmax
        output should be of dimensions (2, ) or (3, ) when draw is being used too.

        :param team_count: (int) number of teams
        :param in_channels: (int) number of input channels
        :param out_channels: (int) number of output channels
        :param conv_out_channels: (int) number of output channels of graph convolutional layer
        :param conv_hidden_channels: (int) number of hidden channels of graph convolutional layer:
        :param activation: (str) activation function (relu, sigmoid, lrelu, tanh), default relu
        :param K: (int)  Chebyshev filter size K (when GCONV_GRU is used), default 1
        :param bias: (bool) whether to use bias
        :param init_ones_: (bool) whether to initialize weights with ones and biases with 0
        :param graph_conv: (str) graph convolutional layer: (GCONV_GRU, GCONV_ELMAN) default GCONV_GRU
        :param normalization: (str) normalization type for GCONV_GRU, ('sym' or 'rw' or None) default sym
        :param discount: (float) discount factor for GCONV_ELMAN, default 0.5
        :param aggr: aggregation type for GCONV_ELMAN, default add
        :param debug: (bool)
        """

        assert K > 0
        assert graph_conv.upper() in ['GCONV_GRU', 'GCONV_ELMAN', ]

        super(RGNN, self).__init__(discount, debug)

        self.team_count = team_count
        self.act_fn = self._activations[activation]

        self.embed_dim = in_channels
        self.embedding = nn.Parameter(torch.full((team_count, in_channels), 1, dtype=torch.float))
        #self.embedding = nn.Embedding(num_embeddings=team_count, embedding_dim=embed_dim)

        # recurrent graph convolution layer
        if graph_conv.upper() == 'GCONV_ELMAN':
            gconv = GConvElman(in_channels, conv_out_channels,
                               aggr=aggr, init_ones_=init_ones_, bias=bias, hidden_channels=conv_hidden_channels)
        else:
            #graph_conv.upper() == 'GCONV_GRU':
            gconv = GConvGRU(in_channels, conv_out_channels, K, bias=bias,
                             init_ones_=init_ones_, normalization=normalization, )

        self.rnn_gconv = gconv

        # simple fully connected part for prediction
        self.pred = nn.Linear(2 * conv_out_channels, out_channels)

        if init_ones_:
            nn.init.eye_(self.pred.weight)
            if self.lin.bias is not None:
                nn.init.zeros_(self.pred.bias)

        self.out_layer = nn.Softmax(dim=0)

    def forward(self, edge_index, home, away, edge_weight=None):
        h = torch.tensor(list(range(self.team_count)))
        h = self.embedding[h].reshape(-1, self.embed_dim)

        # graph convolution
        self._copy_index(edge_index, edge_weight, h)
        h = self.rnn_gconv(h, self.H_edge_index, self.H_edge_weight)

        # prediction
        h = torch.cat([h[away], h[home]], dim=0)
        h = self.pred(h)
        h = self.out_layer(h)
        return h
