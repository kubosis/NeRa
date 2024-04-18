from ._model import *


class BerrarNumerical(BerrarModel):
    """
    Berra with gradient with numerical backward pass via torch autograd system

    Loss function should be MSE, forward pass outputs predicted
    number of goals scored by [home, away] team in Tensor
    """
    def __init__(self, team_count: int, **kwargs):
        """
        :param team_count: number of teams
        :keyword alpha_h: expected number of goals by home team, default = 180
        :keyword beta_h: steepness of exponential for home team, default = 2
        :keyword bias_h:  bias of home team, default = 0
        :keyword alpha_a: expected number of goals by away team, default = 180
        :keyword beta_a: steepness of exponential for away team, default = 2
        :keyword bias_a: bias of away team, default = 0
        """
        super(BerrarNumerical, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        h, a = self.home, self.away = matches

        hatt, hdef = self.att_[h], self.def_[h]
        aatt, adef = self.att_[a], self.def_[a]

        ah, bh, yh = self.alpha_h, self.beta_h, self.bias_h
        aa, ba, ya = self.alpha_a, self.beta_a, self.bias_a

        ghat_h = ah / (1 + torch.exp(-bh * (hatt + adef) - yh))
        ghat_a = aa / (1 + torch.exp(-ba * (aatt + hdef) - ya))

        ghat_h = ghat_h.unsqueeze(0)
        ghat_a = ghat_a.unsqueeze(0)

        return torch.cat((ghat_h, ghat_a), dim=0).view(-1, 1)
