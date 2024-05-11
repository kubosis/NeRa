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

    def forward(self, home, away):
        assert len(home) == len(away) == self.in_channels
        h_half, a_half = len(home) // 2, len(away) // 2

        R_hh, R_ha = home[:h_half], home[h_half:]
        R_ah, R_aa = away[:a_half], away[a_half:]

        R_home = (R_hh + R_ha) / 2
        R_away = (R_aa + R_ah) / 2

        return torch.cat([R_away, R_home], dim=0)


