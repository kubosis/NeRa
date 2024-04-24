from typing import Optional

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU
from ._reccurent_gnn import RecurrentGNN
from torch_geometric.nn import GraphConv, SAGEConv, GCNConv, ChebConv
from ._GConvRNN import GConvElman


class EvalRatingRGNN(RecurrentGNN):
    _activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.2),
    }

    _gconv = {
        "SAGEConv": SAGEConv,
        "GCNConv": GCNConv,
        "GraphConv": GraphConv,
        "ChebConv": ChebConv,
    }

    _rating = {}

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
        rating: Optional[str] = None,
        normalization: str = "sym",
        aggr: str = "add",
        dense_dims: Optional[tuple[int]] = (8, 8, 8, 8, 8),
        conv_dims: tuple[int] = (16, 16, 16),
        dropout_rate: float = 0.1,
        debug=False,
    ):

        assert rgnn_conv.upper() in ["GCONV_GRU", "GCONV_ELMAN", "PEREVERZEVA_RGNN"]
        assert graph_conv in ["GraphConv", "SAGEConv", "GCNConv", "ChebConv"]
        assert rating is None or rating in ["elo", "berrar", "pi"]

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

        # create linear layers if any
        self.linear_layers = None
        if dense_dims is not None:
            self._create_dense_layers()

        self.rating = None

        self.out_layer = nn.Softmax(dim=1)

    def _create_linear_layers(self):
        sequence = [nn.Linear(self.conv_dims[-1], self.dense_dims[0])]
        for i in range(1, len(self.dense_dims)):
            sequence.append(nn.Linear(self.dense_dims[i - 1], self.dense_dims[i]))
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
            sequence.append(GConvElman(self.embed_dim, conv_dims[0]))
            for i in range(1, len(conv_dims)):
                sequence.append(GConvElman(conv_dims[i - 1], conv_dims[i]))
        elif rgnn_conv.upper() == "PEREVERZEVA_RGNN":
            params = self._get_gconv_params(gconv)
            model = self._gconv[gconv]
            model_and_act = nn.Sequential(model(**params), self.activation)
            sequence.append(model_and_act)
            for i in range(1, len(conv_dims)):
                params["in_channels"] = conv_dims[i - 1]
                params["out_channels"] = conv_dims[i]
                model_and_act = nn.Sequential(model(**params), self.activation)
                sequence.append(model_and_act)
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

    def forward(self, edge_index):
        home, away = edge_index

        h = torch.tensor(list(range(self.team_count)))
        h = self.embedding(h).reshape(-1, self.embed_dim)

        for gconv in self.gconv_layers:
            h = gconv(h, edge_index)
            h = self.act_fn(h)

        h = torch.cat([h[home], h[away]], dim=1)
        h = self.dense_sequence(h)
        h = self.out_layer(h)
        return h
