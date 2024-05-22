from ._model import *


class EloManual(EloModel):
    """
    Elo with manual forward and backward pass without gradients

    """

    def __init__(self, team_count: int, **kwargs):
        """
        :param team_count: (int) number of teams
        :keyword default: (float) default rating of all teams, default value = 1000.
        :keyword gamma: (float) impact scale of goal difference, default value = 2.
        :keyword c: (float) rating meta parameter, default value = 3.
        :keyword d: (float) rating meta parameter, default value = 500.
        :keyword k: (float) learning rate, default value = 3.
        """
        super(EloManual, self).__init__(team_count, register=False, **kwargs)
        self.is_manual = True

    def forward(self, matches: Matches):
        self.home, self.away = matches
        home_rating = self.elo[self.home]
        away_rating = self.elo[self.away]

        with torch.no_grad():
            self.E_H = 1 / (
                1 + torch.pow(self.c, ((away_rating - home_rating) / self.d))
            )

        return self.E_H

    def backward(self, result: Result):
        with torch.no_grad():
            home_pts, away_pts = result

            match_outcome = (
                1.0 if home_pts > away_pts else 0.0 if home_pts < away_pts else 1 / 2
            )

            goal_difference = torch.abs(home_pts - away_pts)

            h_i = self.home
            a_i = self.away

            update = (
                self.k * ((1 + goal_difference) ** self.gamma) * (match_outcome - self.E_H)
            )

            self.elo[h_i] += update
            self.elo[a_i] -= update

        return result
