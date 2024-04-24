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

    def __init__(self, in_channels: int, gamma: float = 2., c: float = 3., d: float = 5., hp_grad: bool = False):
        """
        :param gamma: (float) impact scale of goal difference, default value = 2.
        :param c: (float) rating meta parameter, default value = 3.
        :param d: (float) rating meta parameter, default value = 5.
        :param hp_grad: (bool) whether to use gradient descend for ratings hyperparams
        """
        super(Elo, self).__init__()
        self.gamma = gamma
        self.in_channels = in_channels
        if not hp_grad:
            self.c = c
            self.d = torch.full((in_channels,), d, dtype=torch.float)
        else:
            self.c = nn.Parameter(torch.tensor(c), requires_grad=True)
            self.d = nn.Parameter(torch.tensor(torch.full((in_channels,), d, dtype=torch.float)), requires_grad=True)

    def forward(self, home, away):
        assert home.shape == away.shape
        assert len(home) == self.in_channels

        E_H = 1 / (1 + torch.pow(self.c, ((away - home) @ torch.pow(self.d, -1)))).unsqueeze(0)
        E_A = 1 - E_H
        return torch.cat([E_A, E_H], dim=0)
