from typing import Sequence
from collections.abc import Iterable

import torch
import torch.nn as nn
import numpy as np

elo_params = {
    'gamma': float(1.45),
    'c': int(10),
    'd': int(400),
    'k': float(2)
}

Matches = Sequence[np.ndarray]
Result = Sequence[np.ndarray]


class EloModel(nn.Module):
    def __init__(self, team_count: int, default: int = 1000, **kwargs):
        """
        :param team_count: number of teams
        :param default: default rating of all teams
        :keyword gamma: impact scale of goal difference
        :keyword c: rating meta parameter
        :keyword d: rating meta parameter
        :keyword k: learning rate
        """

        assert all(elem in kwargs for elem in elo_params.keys()), "Invalid keyword parameter set"

        super(EloModel, self).__init__()
        for elem in elo_params:
            setattr(self, elem, kwargs[elem])
        self.rating = np.zeros((team_count,)) + default
        self.E_H = None
        self.home, self.away = None, None


class EloManual(EloModel):
    """
    Elo with manual forward and backward pass without gradients
    """
    def __init__(self, team_count: int, **kwargs):
        super(EloManual, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        with torch.no_grad():
            self.home, self.away = matches
            home_rating = self.rating[self.home]
            away_rating = self.rating[self.away]

            self.E_H = 1 / (1 + np.power(self.c, ((away_rating - home_rating) / self.d)))

        return self.E_H

    def backward(self, result: Result):
        with torch.no_grad():
            match_outcome, goal_difference = result

            h_i = self.home
            a_i = self.away

            update = self.k * ((1 + goal_difference) ** self.gamma) * (match_outcome - self.E_H)

            self.rating[h_i] += update
            self.rating[a_i] -= update

        return result


class EloGrad(EloModel):
    """
    Elo with gradient with manual backward pass

    Loss function should be weighted MSE. Weight is the goal difference in the match raised to the power of
    EloGrad.gamma parameter
    """
    def __init__(self, team_count: int, **kwargs):
        super(EloGrad, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        raise NotImplementedError

    def backward(self, grad_output: torch.Tensor):
        raise NotImplementedError


class EloAutoGrad(EloModel):
    """
    Elo with autograd

    Loss function should be weighted MSE. Weight is the goal difference in the match raised to the power of
    EloAutoGrad.gamma parameter
    """
    def __init__(self, team_count: int, **kwargs):
        super(EloAutoGrad, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        raise NotImplementedError
