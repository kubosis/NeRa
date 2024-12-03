import torch.nn as nn
import torch


class Elo(nn.Module):
    """
    Elo with autograd numerical backward pass

    Loss function should be weighted MSE, where Weight is the goal difference in the match raised to the power of
    EloAutoGrad.gamma parameter or CrossEntropyLoss. In case of CrossEntropyLoss the goal difference is
    not accounted for.

    Outputs tensor of same shape as Input

    Forward Computation:
    $E_H = \frac{1}{1 + c^{\frac{a_i - h_i}{d}}}$

    $E_A = 1 - E_H$

    Where:
    - $h_i$: home team rating
    - $a_i$: away team rating
    """

    def __init__(
            self,
            in_channels: int,
            gamma: float = 2.0,
            c: float = 10.0,
            d: float = 400.,
            hp_grad: bool = False,
    ):
        """
        :param gamma: (float) impact scale of goal difference, default value = 2.
        :param c: (float) rating meta parameter, default value = 3.
        :param d: (float) rating meta parameter, default value = 0.8.
        :param hp_grad: (bool) whether to use gradient descend for ratings hyperparams
        """
        assert in_channels > 0

        super(Elo, self).__init__()
        self.gamma = gamma
        self.in_channels = in_channels
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float), requires_grad=hp_grad)
        self.d = nn.Parameter(torch.tensor(d, dtype=torch.float), requires_grad=hp_grad)

    def forward(self, home, away):
        assert home.shape == away.shape
        assert len(home) == self.in_channels

        E_H = 1 / (1 + torch.pow(self.c, ((away - home) / self.d)))

        E_A = 1 - E_H
        return torch.cat([E_A, E_H], dim=0)
