from ._model import *


class BerrarNumerical(BerrarModel):
    def __init__(self, team_count: int, **kwargs):
        super(BerrarNumerical, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        self.home, self.away = matches
        E_H = 1 / (1 + torch.pow(self.c, ((self.elo[self.away] - self.elo[self.home]) / self.d)))
        return E_H
