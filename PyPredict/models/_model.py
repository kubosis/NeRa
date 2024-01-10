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

