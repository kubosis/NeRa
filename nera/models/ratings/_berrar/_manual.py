from ._model import *


class BerrarManual(BerrarModel):
    """
    Berrar with manual forward and backward pass without gradients

    """

    def __init__(self, team_count: int, **kwargs):
        """
        :param team_count: number of teams
        :param team_count: number of teams
        :keyword alpha_h: expected number of goals by home team, default = 180
        :keyword beta_h: steepness of exponential for home team, default = 2
        :keyword bias_h:  bias of home team, default = 0
        :keyword alpha_a: expected number of goals by away team, default = 180
        :keyword beta_a: steepness of exponential for away team, default = 2
        :keyword bias_a: bias of away team, default = 0
        :keyword lr_h_att: home team's attack strength learning rate, default = 0.1
        :keyword lr_a_att: away team's attack strength learning rate, default = 0.1
        :keyword lr_h_def: home team's defensive weakness learning rate, default = 0.1
        :keyword lr_a_def: away team's defensive weakness learning rate, default = 0.1
        """
        super(BerrarManual, self).__init__(team_count, **kwargs)
        self.is_manual = True

        self.att_.requires_grad = False
        self.def_.requires_grad = False

    def forward(self, matches: Matches):
        h, a = self.home, self.away = matches

        hatt, hdef = self.att_[h], self.def_[h]
        aatt, adef = self.att_[a], self.def_[a]

        alpha_a, alpha_h = self.alpha_a, self.alpha_h
        beta_a, beta_h = self.beta_a, self.beta_h
        bias_a, bias_h = self.bias_a, self.bias_h

        self.g_h = alpha_h / (1 + torch.exp(-beta_h * (hatt + adef) - bias_h))
        self.g_a = alpha_a / (1 + torch.exp(-beta_a * (aatt + hdef) - bias_a))

        out = torch.tensor([self.g_h, self.g_a])
        return out

    def backward(self, result: Result):
        goals_home, goals_away = result

        h, a = self.home, self.away

        lr_h_att, lr_h_def = self.lr_h_att, self.lr_h_def
        lr_a_att, lr_a_def = self.lr_a_att, self.lr_a_def

        self.att_[h] += lr_h_att * (goals_home - self.g_h)
        self.def_[h] += lr_h_def * (goals_away - self.g_a)

        self.att_[a] += lr_a_att * (goals_away - self.g_a)
        self.def_[a] += lr_a_def * (goals_home - self.g_h)

        return result
