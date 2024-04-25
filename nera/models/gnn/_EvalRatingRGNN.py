from typing import Optional

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.nn import GraphConv, SAGEConv, GCNConv, ChebConv

from ._GConvElman import GConvElman
from ._reccurent_gnn import RecurrentGNN
from ..ratings import Elo, Berrar, Pi


class EvalRatingRGNN(RecurrentGNN):
    _activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "lrelu": nn.LeakyReLU(0.2),
    }

    _gconv = {
        "SAGEConv": SAGEConv,
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
            dense_dims: Optional[tuple[int]] = (8, 8, 8, 8, 8),
            conv_dims: tuple[int] = (16, 16, 16),
            dropout_rate: float = 0.1,
            debug=False,
            **rating_kwargs
    ):

        assert rgnn_conv.upper() in ["GCONV_GRU", "GCONV_ELMAN", "PEREVERZEVA_RGNN"]
        assert graph_conv in ["GraphConv", "GCNConv", "ChebConv"]
        assert rating in ["elo", "berrar", "pi"]

        assert conv_dims is not None
        assert conv_dims[-1] % 2 == 0 or rating == "elo"  # since berrar and pi needs 2*n values for each team

        super(EvalRatingRGNN, self).__init__(discount, debug, correction)

        self.team_count = team_count

        self.embed_dim = embed_dim
        self.target_dim = target_dim
        self.rating_dim = conv_dims[-1]
        self.conv_dims = conv_dims
        self.dense_dims = dense_dims

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
            self.out_layer = nn.Softmax(dim=0)

    def _resolve_rating(self, rating: Optional[str], **rating_kwargs):
        in_channels = self.conv_dims[-1] if self.dense_dims is not None else self.conv_dims[-1]
        rtg = self._rating[rating](in_channels=in_channels, **rating_kwargs)
        return rtg

    def _create_linear_layers(self):
        sequence = []
        if self.dense_dims is not None:
            in_channels = self.conv_dims[-1] * 2 if self.rating_str == "elo" else self.conv_dims[-1]
            sequence.append(nn.Linear(in_channels, self.dense_dims[0]))
            for i in range(1, len(self.dense_dims)):
                sequence.append(nn.Linear(self.dense_dims[i - 1], self.dense_dims[i]))
            if self.dense_dims[-1] != self.target_dim:
                sequence.append(nn.Linear(self.dense_dims[-1], self.target_dim))
        else:
            if (self.rating_str == "elo") and (self.conv_dims[-1] * 2 != self.target_dim):
                sequence.append(nn.Linear(self.conv_dims[-1] * 2, self.target_dim))
            elif (self.rating_str in ["berrar", "pi"]) and (self.conv_dims[-1] != self.target_dim):
                sequence.append(nn.Linear(self.conv_dims[-1], self.target_dim))
        self.linear_layers = sequence

    def _create_rgnn_layers(
            self, rgnn_conv: str, K: int, normalization: str, gconv: str
    ):
        conv_dims = self.conv_dims
        sequence = []
        if rgnn_conv.upper() == "GCONV_GRU":
            sequence.append(
                GConvGRU(self.embed_dim, conv_dims[0], K, normalization=normalization)
            )
            for i in range(1, len(conv_dims)):
                sequence.append(
                    GConvGRU(
                        conv_dims[i - 1], conv_dims[i], K, normalization=normalization
                    )
                )
        elif rgnn_conv.upper() == "GCONV_ELMAN":
            sequence.append(GConvElman(self.embed_dim, conv_dims[0], gconv, K=self.K, normalization=normalization))
            for i in range(1, len(conv_dims)):
                sequence.append(
                    GConvElman(conv_dims[i - 1], conv_dims[i], gconv, K=self.K, normalization=normalization))
        elif rgnn_conv.upper() == "PEREVERZEVA_RGNN":
            params = self._get_gconv_params(gconv)
            model = self._gconv[gconv]
            sequence.append(model(**params))
            for i in range(1, len(conv_dims)):
                params["in_channels"] = conv_dims[i - 1]
                params["out_channels"] = conv_dims[i]
                sequence.append(model(**params))
        self.gconv_layers = sequence

    def _get_gconv_params(self, gconv: str):
        _model_params = {
            "in_channels": self.embed_dim,
            "out_channels": self.conv_dims[0],
        }
        if gconv == "GraphConv" or gconv == "SAGEConv":
            # for our purposes we really just consider add
            _model_params["aggr"] = self.aggr
        if gconv == "ChebConv":
            _model_params["normalization"] = self.normalization
            _model_params["K"] = self.K
        return _model_params

    def forward(self, edge_index, home, away, edge_weight=None):
        # get teams embedding
        h = torch.tensor(list(range(self.team_count)))
        h = self.embedding(h).reshape(-1, self.embed_dim)

        # graph convolution
        self._copy_index(edge_index, edge_weight, h)
        for gconv in self.gconv_layers:
            h = gconv(h, self.H_edge_index, self.H_edge_weight)

        # rating
        h_rtg, a_rtg = h[home], h[away]
        h = self.rating(h_rtg, a_rtg)

        # linear layers if any
        for lin in self.linear_layers:
            h = lin(h)
            h = self.activation(h)
            h = self.dropout(h)

        h = self.out_layer(h)
        return h
