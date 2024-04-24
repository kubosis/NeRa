import copy
from typing import Optional

import torch
import torch.nn as nn

from ._reccurent_gnn import RecurrentGNN
from ._gconv_gru_copy import GConvGRU


#from torch_geometric_temporal.nn import GConvGRU


class RatingGNN(RecurrentGNN):
    _activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'lrelu': nn.LeakyReLU(0.2),
        'tanh': nn.Tanh(),
    }

    def __init__(self, team_count: int,
                 embed_dim: int,
                 out_channels: int,
                 rnn_gconv: nn.Module,
                 rating: nn.Module,
                 discount: float = 0.5,
                 default: float = 1.0,
                 correction: bool = False,
                 debug: bool = False,
                 ):

        """
        Baseline recurrent graph neural network for match outcome prediction. The architecture is following:
        signal x (match home vs away team)
        h = embedding(teams)
        recurrent graph convolutional layer
        away = embeddings[away] home = embeddings[home]
        out = rating(away, home)
        output should be of dimensions (2, ) or (3, ) when draw is being used too.

        :param team_count: (int) number of teams
        :param in_channels: (int) number of input channels
        :param out_channels: (int) number of output channels

        """

        super(RatingGNN, self).__init__(discount, debug, correction)

        self.H_edge_index = None
        self.H_edge_weight = None

        self.team_count = team_count

        self.embed_dim = embed_dim
        #self.embedding = nn.Parameter(torch.full((team_count, embed_dim), default, dtype=torch.float))
        self.embedding = nn.Embedding(num_embeddings=team_count, embedding_dim=embed_dim, dtype=torch.float)
        nn.init.ones_(self.embedding.weight)

        # recurrent graph convolution layer
        self.rnn_gconv = rnn_gconv

        # rating layer
        self.pred = rating

    def forward(self, edge_index, home, away, edge_weight=None):
        h = torch.tensor(list(range(self.team_count)))
        h = self.embedding(h).reshape(-1, self.embed_dim)

        # graph convolution
        self._copy_index(edge_index, edge_weight, h)
        h = self.rnn_gconv(h, self.H_edge_index, self.H_edge_weight)

        # prediction
        home, away = h[home], h[away]
        h = self.pred(home, away)

        return h
