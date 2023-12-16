from abc import ABC
from abc import abstractmethod

import pandas as pd
from torch.nn import ReLU, Tanh, LeakyReLU, ModuleList, Dropout, LogSoftmax
import torch
from torch_geometric.nn import GraphConv

activations = {
    'relu': ReLU(),
    'tanh': Tanh(),
    'leaky': LeakyReLU(0.2)
}


class GeneralModel(ABC):
    def __init__(self, df: pd.DataFrame, num_teams: int):
        ABC.__init__(self)
        self.df = df
        self.num_teams = num_teams

    @abstractmethod
    def compute_model(self) -> None:
        """
        compute model from self.dataframe
        """


class GeneralGNNModel(GeneralModel, ABC, torch.nn.Module):
    def __init__(self, df: pd.DataFrame, num_teams, *args, embed_dim=10,
                 n_conv=3, conv_dims=(32, 32, 32, 16), n_dense=5, dense_dims=(8, 8, 8, 8, 8),
                 activ_fn='leaky', target_dim=3, log_softmax_dim=1, dropout=0.1, **kwargs):
        torch.nn.Module.__init__(self, *args, **kwargs)
        GeneralModel.__init__(self, df, num_teams)

        # model hyper params
        self.embed_dim = embed_dim
        self.n_conv = n_conv
        self.conv_dims = conv_dims
        self.n_dense = n_dense
        self.activation = activations[activ_fn]

        conv_layers = [GraphConv(self.embed_dim, self.conv_dims[0])]
        for i in range(n_conv - 1):
            conv_layers.append(GraphConv(conv_dims[i], conv_dims[i + 1]))
        self.conv_layers = ModuleList(conv_layers)

        lin_layers = [torch.nn.Linear(conv_dims[n_conv - 1] * 2, dense_dims[0])]
        for i in range(n_dense - 2):
            lin_layers.append(torch.nn.Linear(dense_dims[i], dense_dims[i + 1]))
        lin_layers.append(torch.nn.Linear(dense_dims[n_dense - 2], target_dim))

        self.lin_layers = ModuleList(lin_layers)

        self.out = LogSoftmax(dim=log_softmax_dim)
        self.drop = Dropout(p=dropout)

    @abstractmethod
    def forward(self, data, home, away):
        ...
