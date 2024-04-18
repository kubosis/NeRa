from ._model import *


class EloNumerical(EloModel):
    """
    Elo with autograd numerical backward pass

    Loss function should be weighted MSE. Weight is the goal difference in the match raised to the power of
    EloAutoGrad.gamma parameter
    """

    def __init__(self, team_count: int, **kwargs):
        """
        :param team_count: (int) number of teams
        :keyword default: (float) default rating of all teams, default value = 1000.
        :keyword gamma: (float) impact scale of goal difference, default value = 2.
        :keyword c: (float) rating meta parameter, default value = 3.
        :keyword d: (float) rating meta parameter, default value = 500.
        """
        super(EloNumerical, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        self.home, self.away = matches
        E_H = 1 / (1 + torch.pow(self.c, ((self.elo[self.away] - self.elo[self.home]) / self.d)))
        return E_H
