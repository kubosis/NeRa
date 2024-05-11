from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, A3TGCN
from torch_geometric.nn import GraphConv, SAGEConv, GCNConv, ChebConv

from ._GConvElman import GConvElman
from ._reccurent_gnn import RecurrentGNN
from ..ratings import Elo, Berrar, Pi


class RatingRGNN(RecurrentGNN):
    _activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "lrelu": nn.LeakyReLU(0.2),
        "sigmoid": nn.Sigmoid(),
    }

    _gconv = {
        "GCNConv": GCNConv,
        "GraphConv": GraphConv,
        "ChebConv": ChebConv,
    }

    _rating = {
        "elo": Elo,
        "berrar": Berrar,
        "pi": Pi,
    }

    def __init__(
            self,
            team_count: int,
            embed_dim: int,
            target_dim: int,
            discount: float = 0.8,
            correction: bool = False,
            activation: str = "relu",
            K: int = 2,
            rgnn_conv: str = "GCONV_GRU",
            graph_conv: str = "GCNConv",
            rating: str = None,
            normalization: Optional[str] = "sym",
            aggr: str = "add",
            dense_layers: int = 1,
            conv_layers: int = 1,
            dropout_rate: float = 0.1,
            debug=False,
            **rating_kwargs
    ):

        assert rgnn_conv.upper() in ["GCONV_GRU", "GCONV_ELMAN"]
        assert graph_conv in ["GraphConv", "GCNConv", "ChebConv"]
        assert rating in ["elo", "berrar", "pi", None]

        assert embed_dim % 2 == 0 or rating == "elo"  # since berrar and pi needs 2*n values for each team
        assert conv_layers > 0 and dense_layers >= 0

        super(RatingRGNN, self).__init__(discount, debug, correction)

        self.team_count = team_count

        self.embed_dim = embed_dim
        self.target_dim = target_dim
        self.rating_dim = embed_dim
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

        self.K = K
        self.normalization = normalization
        self.aggr = aggr

        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = self._activations[activation]
        self.embedding = nn.Embedding(
            num_embeddings=team_count, embedding_dim=embed_dim
        )

        # create graph convolution recurrent layers
        self.gconv_layers = None
        self._create_rgnn_layers(rgnn_conv, K, normalization, graph_conv)

        # create rating layer
        self.rating_str = rating
        self.rating = self._resolve_rating(rating, **rating_kwargs)

        # create linear layers if any
        self.linear_layers = None
        self._create_linear_layers()

        self.out_layer = None
        self._resolve_out_layer()

    def _resolve_out_layer(self):
        if self.rating_str == "elo" and self.linear_layers == []:
            # since elo already gives probability no need to use softmax
            self.out_layer = nn.Identity(self.target_dim)
        else:
            self.out_layer = RescaleByMax()

    def _resolve_rating(self, rating: Optional[str], **rating_kwargs):
        in_channels = self.rating_dim
        if rating is not None:
            rtg = self._rating[rating](in_channels=in_channels, **rating_kwargs)
        else:
            rtg = None
        return rtg

    def _create_linear_layers(self):
        sequence = []
        in_channels = 2 * self.rating_dim if self.rating_str in ["elo", None] else self.rating_dim
        if self.dense_layers > 0:
            sequence.append(nn.Linear(in_channels, self.rating_dim))
            for i in range(1, self.dense_layers):
                sequence.append(nn.Linear(self.rating_dim, self.rating_dim))
            if self.rating_dim != self.target_dim:
                sequence.append(nn.Linear(self.rating_dim, self.target_dim))
        else:
            if in_channels != self.target_dim:
                sequence.append(nn.Linear(in_channels, self.target_dim))
        self.linear_layers = sequence

    def _create_rgnn_layers(
            self, rgnn_conv: str, K: int, normalization: str, gconv: str
    ):
        sequence = []
        if rgnn_conv.upper() == "GCONV_GRU":
            m = GConvGRU(self.embed_dim, self.rating_dim, K, normalization=normalization)
            m.to(self.device)
            sequence.append(m)
            for i in range(1, self.conv_layers):
                m = GConvGRU(
                        self.rating_dim, self.rating_dim, K, normalization=normalization
                    )
                m.to(self.device)
                sequence.append(m)
        elif rgnn_conv.upper() == "GCONV_ELMAN":
            sequence.append(GConvElman(self.embed_dim, self.rating_dim, gconv, K=self.K, normalization=normalization))
            for i in range(1, self.conv_layers):
                sequence.append(
                    GConvElman(self.rating_dim, self.rating_dim, gconv, K=self.K, normalization=normalization))
        elif rgnn_conv.upper() == "PEREVERZEVA_RGNN":
            params = self._get_gconv_params(gconv)
            model = self._gconv[gconv]
            sequence.append(model(**params))
            params["in_channels"] = self.rating_dim
            for i in range(1, self.conv_layers):
                m = model(**params)
                m.to(self.device)
                sequence.append(m)
        self.gconv_layers = sequence

    def _get_gconv_params(self, gconv: str):
        _model_params = {
            "in_channels": self.embed_dim,
            "out_channels": self.rating_dim,
        }
        if gconv == "GraphConv":
            # for our purposes we really just consider add
            _model_params["aggr"] = self.aggr
        if gconv == "ChebConv":
            _model_params["normalization"] = self.normalization
            _model_params["K"] = self.K
        return _model_params

    def forward(self, edge_index, home, away, edge_weight=None, home_goals=None, away_goals=None):
        # get teams embedding
        h = torch.tensor(list(range(self.team_count))).to(self.device)
        h = self.embedding(h).reshape(-1, self.embed_dim)

        # graph convolution
        self._copy_index(edge_index, edge_weight, h)
        for gconv in self.gconv_layers:
            h = gconv(h, self.H_edge_index, self.H_edge_weight)

        # rating
        h_rtg, a_rtg = h[home], h[away]
        if self.rating_str is None:
            # no rating layer, just MLP prediction
            h = torch.cat([a_rtg, h_rtg], dim=0)
        else:
            h = self.rating(h_rtg, a_rtg)

        # linear layers if any
        i = 0
        for lin in self.linear_layers:
            i += 1
            h = lin(h)
            h = self.activation(h)
            # no dropout before output
            h = self.dropout(h) if i != len(self.linear_layers) else h

        h = self.out_layer(h)
        return h


class RescaleByMax(nn.Module):
    def forward(self, x):
        max_value = F.normalize(x, p=float('inf'), dim=0)
        return x / max_value
