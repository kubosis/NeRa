import torch
from torch.nn import LogSoftmax, ReLU, Tanh, LeakyReLU, ModuleList, Dropout
from torch_geometric.nn import GCNConv, GraphConv, ChebConv

from ._model import GeneralGNNModel


class GNNModel(GeneralGNNModel):
    ...
