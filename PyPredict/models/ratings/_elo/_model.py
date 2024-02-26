from typing import Sequence

import torch
import torch.nn as nn
import numpy as np

elo_params = {
    'k': torch.tensor(3., dtype=torch.float64),
    'gamma': torch.tensor(2., dtype=torch.float64),
    'c': torch.tensor(3., dtype=torch.float64),
    'd': torch.tensor(500., dtype=torch.float64),
}

Matches = Sequence[np.ndarray]
Result = Sequence[np.ndarray]
Index = int


class EloModel(nn.Module):
    def __init__(self, team_count: int, default: float = 1000., **kwargs):
        """
        :param team_count: number of teams
        :param default: default rating of all teams
        :keyword gamma: impact scale of goal difference
        :keyword c: rating meta parameter
        :keyword d: rating meta parameter
        :keyword k: learning rate
        """

        super(EloModel, self).__init__()
        for elem in elo_params:
            if elem in kwargs:
                setattr(self, elem, kwargs[elem])
            else:
                setattr(self, elem, elo_params[elem])

        assert (isinstance(self.c, torch.Tensor) and isinstance(self.d, torch.Tensor))

        self.c = self.c.detach()
        self.d = self.d.detach()

        if 'cd_grad' in kwargs:
            cd_grad = kwargs['cd_grad']
            if cd_grad:
                self.c.requires_grad = True
                self.d.requires_grad = True
                self.c = nn.Parameter(self.c)
                self.d = nn.Parameter(self.d)

        self.rating = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.E_H = None
        self.home, self.away = None, None
