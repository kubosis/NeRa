from ._model import *


class PiManual(PiModel):
    def __init__(self, team_count: int, **kwargs):
        super(PiManual, self).__init__(team_count, **kwargs)
        self.is_manual = True

        self.home_rating.requires_grad = False
        self.away_rating.requires_grad = False

        self.ghat_H = None
        self.ghat_A = None

    def forward(self, matches: Matches):
        h, a = self.home, self.away = matches

        rh_H, rh_A = self.home_rating[h], self.home_rating[a]
        ra_H, ra_A = self.away_rating[h], self.away_rating[a]

        c = self.c

        gda_H = torch.pow(10, torch.abs(ra_H) / c) - 1
        gdh_H = torch.pow(10, torch.abs(rh_H) / c) - 1
        gda_A = torch.pow(10, torch.abs(ra_A) / c) - 1
        gdh_A = torch.pow(10, torch.abs(rh_A) / c) - 1

        self.ghat_H = gdh_H - gda_H
        self.ghat_A = gdh_A - gda_A

        goal_diff = self.ghat_H - self.ghat_A
        pred_home_win = 1 / (1 + torch.exp(-goal_diff))
        return pred_home_win

    def backward(self, result: Result):
        goals_home, goals_away = result

        h, a = self.home, self.away

        err_home = torch.abs(goals_home - self.ghat_H)
        err_away = torch.abs(goals_away - self.ghat_A)

        def psi(c, e):
            return c * torch.log10(1 + e)

        self.home_rating[h] += self.lambda_ * psi(self.c, err_home)
        self.away_rating[h] += self.gamma * (self.lambda_ * psi(self.c, err_home))

        self.home_rating[a] += self.lambda_ * psi(self.c, err_away)
        self.away_rating[a] += self.gamma * (self.lambda_ * psi(self.c, err_away))

        # print(self.home_rating[h], self.away_rating[h], self.home_rating[a], self.away_rating[a])
        return result
