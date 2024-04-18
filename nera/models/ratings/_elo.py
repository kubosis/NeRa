import torch.nn as nn
import torch


class Elo(nn.Module):
    """
    Elo with autograd numerical backward pass (for development details see _standalone/_elo/*.py)

    Loss function should be weighted MSE. Weight is the goal difference in the match raised to the power of
    EloAutoGrad.gamma parameter or CrossEntropyLoss. In case of CrossEntropyLoss the goal difference is
    not accounted for.

    Outputs tensor of same shape as Input
    """

    def __init__(self, gamma: float = 2., c: float = 3., d: float = 5., hp_grad: bool = False):
        """
        :param gamma: (float) impact scale of goal difference, default value = 2.
        :param c: (float) rating meta parameter, default value = 3.
        :param d: (float) rating meta parameter, default value = 5.
        :param hp_grad: (bool) whether to use gradient descend for ratings hyperparams
        """
        super(Elo, self).__init__()
        self.gamma = gamma
        if not hp_grad:
            self.c = c
            self.d = d
        else:
            self.c = nn.Parameter(torch.tensor(c), requires_grad=True)
            self.d = nn.Parameter(torch.tensor(d), requires_grad=True)

    def forward(self, home_elo, away_elo):
        E_H = 1 / (1 + torch.pow(self.c, ((away_elo - home_elo) / self.d)))
        E_A = 1 - E_H
        return torch.cat([E_H, E_A], dim=0)
