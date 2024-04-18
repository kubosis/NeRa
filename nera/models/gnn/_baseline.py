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
                 in_channels: int = 20,
                 out_channels: int = 3,
                 conv_out_channels: int = 1,
                 conv_hidden_channels: int = 1,
                 dropout_rate: float = 0.1,
                 activation: str = 'relu',
                 K: int = 1,
                 bias: bool = True,
                 init_ones_: bool = False,
                 graph_conv: str = 'GCONV_GRU',
                 normalization: Optional[str] = 'sym',
                 discount: float = 0.5,
                 aggr: str = 'add',
                 debug=False):

        assert K > 0
        assert graph_conv.upper() in ['GCONV_GRU', 'GCONV_ELMAN', ]
        assert 0 <= discount <= 1

        super(RGNN, self).__init__()

        self.discount = discount
        self.H_edge_index = None
        self.H_edge_weight = None

        self.last_edge = {}

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

        self.debug = debug
        self.embedding_progression = []
        self.H_progression = []

        # simple fully connected part for prediction
        self.lin = nn.Linear(2 * conv_out_channels, out_channels)

        if init_ones_:
            nn.init.eye_(self.lin.weight)
            if self.lin.bias is not None:
                nn.init.zeros_(self.lin.bias)

        self.out_layer = nn.Softmax(dim=0)

    def _correct_weight(self, edge_index, edge_weight) -> None:
        t1, t2 = edge_index[0]
        t1, t2 = t1.item(), t2.item()
        i_t1 = self.last_edge[t1] if t1 in self.last_edge else None
        i_t2 = self.last_edge[t2] if t2 in self.last_edge else None

        w2, w1 = edge_weight

        if i_t1 is not None:
            # edge to t1 exists in the graph before this match
            last_w1 = self.H_edge_weight[i_t1]
            if (w1 < 0 and last_w1 < 0) or (w1 > 0 and last_w1 < 0):
                # (now L, last L) or (now W lat L) -> needs correction
                self.H_edge_weight[i_t1] *= -1.

        if i_t2 is not None:
            last_w2 = self.H_edge_weight[i_t2]
            if (w2 < 0 and last_w2 < 0) or (w2 > 0 and last_w2 < 0):
                # (now L, last L) or (now W lat L) -> needs correction
                self.H_edge_weight[i_t2] *= -1.

    def _copy_index(self, edge_index, edge_weight):
        if edge_weight is None:
            new_edge_weight = torch.ones_like(edge_index[0, :]).detach().to(torch.float)
        else:
            new_edge_weight = edge_weight.detach().clone().to(torch.float)
        if self.H_edge_weight is None:
            self.H_edge_weight = new_edge_weight
        else:
            self.H_edge_weight *= self.discount
            # self._correct_weight(edge_index, edge_weight)
            self.H_edge_weight[-2:] = torch.abs(self.H_edge_weight[-2:])
            self.H_edge_weight = torch.cat([self.H_edge_weight, new_edge_weight])

        new_edge_index = edge_index.detach().clone()
        if self.H_edge_index is None:
            self.H_edge_index = new_edge_index
        else:
            self.H_edge_index = torch.cat([self.H_edge_index, new_edge_index], dim=1)

        # save edges for later weight correction // we save index of last incoming edge
        i1 = self.H_edge_index.shape[1] - 1
        i2 = self.H_edge_index.shape[1] - 2
        self.last_edge[new_edge_index[0, 0].item()] = i1
        self.last_edge[new_edge_index[0, 1].item()] = i2

    def reset_index(self):
        self.H_edge_index = None
        self.H_edge_weight = None

    def forward(self, edge_index, home, away, edge_weight=None):
        h = torch.tensor(list(range(self.team_count)))
        h = self.embedding[h].reshape(-1, self.embed_dim)
        if self.debug:
            self.embedding_progression.append(h.detach().clone().numpy())

        # graph convolution
        self._copy_index(edge_index, edge_weight)
        h = self.rnn_gconv(h, self.H_edge_index, self.H_edge_weight)
        #h = self.rnn_gconv(h, edge_index, edge_weight)

        # prediction
        h = torch.cat([h[away], h[home]], dim=0)
        h = self.lin(h)
        h = self.out_layer(h)
        return h
