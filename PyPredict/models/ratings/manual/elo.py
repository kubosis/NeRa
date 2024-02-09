from typing import Sequence

import torch
import torch.nn as nn
import numpy as np

elo_params = {
    'gamma': float(1.45),
    'c': int(10),
    'd': int(400),
    'k': float(2)
}

Matches = Sequence[np.ndarray, np.ndarray]
Match_outcome = np.ndarray  # 1 = home win, 1/2 draw, 0 = away win
Goal_difference = np.ndarray
Result: Sequence[Match_outcome, Goal_difference]


class EloManual(nn.Module):
    def __init__(self, team_count: int, default: int = 1000, **kwargs):
        """
        :param team_count: number of teams
        :keyword gamma: impact scale of goal difference
        :keyword c: rating meta parameter
        :keyword d: rating meta parameter
        :keyword k: learning rate
        """

        assert all(elem in kwargs for elem in elo_params.keys()), "Invalid keyword parameter set"

        super(EloManual, self).__init__()
        for elem in elo_params:
            setattr(self, elem, kwargs[elem])
        self.rating = np.zeros((team_count,)) + default
        self.E_H = self.E_A = 0
        self.home, self.away = None, None

    def forward(self, matches: Matches):
        with torch.no_grad:
            self.home, self.away = matches
            home_rating = self.rating[self.home]
            away_rating = self.rating[self.away]

            self.E_H = 1 / (1 + np.power(self.c, (home_rating - away_rating) / self.d))

        return self.E_H

    def backward(self, result: Result):
        with torch.no_grad:
            match_outcome, goal_difference = result
            scale = self.k * ((1 + goal_difference)**self.gamma)
            self.rating[self.home] += scale * (match_outcome - self.E_H)
            self.rating[self.away] -= scale * (match_outcome - self.E_H)

        return result
