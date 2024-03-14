from ._model import *


class BerrarNumerical(BerrarModel):
    """
    Berra with gradient with numerical backward pass via torch autograd system

    Loss function should be MSE, forward pass outputs predicted
    number of goals scored by [home, away] team in Tensor
    """
    def __init__(self, team_count: int, **kwargs):
        """
        :param team_count: number of teams
        :keyword alpha_h: expected number of goals by home team, default = 180
        :keyword beta_h: steepness of exponential for home team, default = 2
        :keyword bias_h:  bias of home team, default = 0
        :keyword alpha_a: expected number of goals by away team, default = 180
        :keyword beta_a: steepness of exponential for away team, default = 2
        :keyword bias_a: bias of away team, default = 0
        """
        super(BerrarNumerical, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        self.home, self.away = matches
        E_H = 1 / (1 + torch.pow(self.c, ((self.elo[self.away] - self.elo[self.home]) / self.d)))
        return E_H
