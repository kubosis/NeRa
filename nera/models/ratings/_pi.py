import copy

import torch.nn as nn
import torch


class Pi(nn.Module):
    def __init__(self,
                 in_channels: int,
                 lambda_: float = 1.,
                 gamma: float = 1.,
                 c: float = 1.,
                 hp_grad: bool = False,
                 ):
        assert in_channels % 2 == 0 and in_channels > 0
        super(Pi, self).__init__()

        self.in_channels = in_channels

        if not hp_grad:
            self.c = c
            self.lambda_ = lambda_
            self.gamma = gamma
        else:
            self.c = nn.Parameter(torch.tensor(c, dtype=torch.float), requires_grad=True)
            self.lambda_ = nn.Parameter(torch.tensor(lambda_, dtype=torch.float), requires_grad=True)
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=True)

    def forward(self, home, away, home_goals=None, away_goals=None):
        assert len(home) == len(away) == self.in_channels

        h_half, a_half = len(home) // 2, len(away) // 2

        if home_goals is not None and away_goals is not None:
            # only during training, not in validation
            R_hht, R_hat = home[:h_half], home[h_half:]
            R_aht, R_aat = away[:a_half], away[a_half:]

            ghat_da_home = torch.pow(10, torch.abs(R_hat) / self.c) - 1
            ghat_dh_home = torch.pow(10, torch.abs(R_hht) / self.c) - 1
            ghat_da_away = torch.pow(10, torch.abs(R_aat) / self.c) - 1
            ghat_dh_away = torch.pow(10, torch.abs(R_aht) / self.c) - 1

            ghat_d_home = ghat_dh_home - ghat_da_home
            ghat_d_away = ghat_dh_away - ghat_da_away

            error_home = torch.abs(ghat_d_home - home_goals)
            error_away = torch.abs(ghat_d_away - away_goals)

            last_R_hh = R_aht.clone()
            R_hh = R_hht + self.lambda_ * self.c * torch.log10(error_home)
            R_ha = R_hat + self.gamma * (R_hh - last_R_hh)

            last_R_ah = R_aht.clone()
            R_ah = R_aht + self.lambda_ * self.c * torch.log10(error_away)
            R_aa = R_aat + self.gamma * (R_ah - last_R_ah)
        else:
            R_hh, R_ha = home[:h_half], home[h_half:]
            R_ah, R_aa = away[:a_half], away[a_half:]

        R_home = (R_hh + R_ha) / 2
        R_away = (R_aa + R_ah) / 2

        return torch.cat([R_away, R_home], dim=0)


