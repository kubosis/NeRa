from ._model import *


class BerrarManual(BerrarModel):
    def __init__(self, team_count: int, **kwargs):
        super(BerrarManual, self).__init__(team_count, **kwargs)
        self.is_manual = True

    def forward(self, matches: Matches):
        h, a = self.home, self.away = matches

        h_att, h_def = self.h_att[h], self.h_def[h]
        a_att, a_def = self.a_att[a], self.a_def[a]

        alpha_a, alpha_h = self.alpha_a[a], self.alpha_h[h]
        beta_a, beta_h = self.beta_a[a], self.beta_h[h]
        bias_a, bias_h = self.bias_a[a], self.bias_h[h]

        with torch.no_grad():
            self.g_h = alpha_h / (1 + torch.exp(-beta_h * (h_att + a_def) - bias_h))
            self.g_a = alpha_a / (1 + torch.exp(-beta_a * (a_att + h_def) - bias_a))

        return self.g_h, self.g_a

    def backward(self, result: Result):
        goals_home, goals_away = result

        h, a = self.home, self.away

        lr_h_att, lr_h_def = self.lr_h_att[h], self.ler_h_def[h]
        lr_a_att, lr_a_def = self.lr_a_att[a], self.lr_a_def[a]

        with torch.no_grad():
            self.h_att[h] += lr_h_att * (goals_home - self.g_h)
            self.h_def[h] += lr_h_def * (goals_away - self.g_a)

            self.a_att[a] += lr_a_att * (goals_away - self.g_a)
            self.a_def[a] += lr_a_def * (goals_home - self.g_h)

        return result


