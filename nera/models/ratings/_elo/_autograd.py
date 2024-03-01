from ._model import *


class EloAutoGrad(EloModel):
    """
    Elo with autograd

    Loss function should be weighted MSE. Weight is the goal difference in the match raised to the power of
    EloAutoGrad.gamma parameter
    """

    def __init__(self, team_count: int, **kwargs):
        """
        :param team_count: (int) number of teams
        :keyword default: (float) default rating of all teams, default value = 1000.
        :keyword gamma: (float) impact scale of goal difference, default value = 2.
        :keyword c: (1, torch.Tensor, float64) rating meta parameter, default value = 3.
        :keyword d: (1, torch.Tensor, float64) rating meta parameter, default value = 500.
        :keyword k: (float) learning rate, default value = 2.
        """
        super(EloAutoGrad, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        self.home, self.away = matches
        E_H = 1 / (1 + torch.pow(self.c, ((self.elo[self.away] - self.elo[self.home]) / self.d)))
        return E_H
