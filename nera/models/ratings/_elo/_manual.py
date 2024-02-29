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
        :keyword k: (float) learning rate, default value = 2.
        """
        super(EloManual, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        self.home, self.away = matches
        home_rating = self.rating[self.home]
        away_rating = self.rating[self.away]

        with torch.no_grad():
            self.E_H = 1 / (1 + torch.pow(self.c, ((away_rating - home_rating) / self.d)))

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
