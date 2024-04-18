import torch
from torch import Tensor

from ._model import *


class EloSymbolical(EloModel):
    """
    Symbolic Elo Model
    """

    def __init__(self, team_count: int, rating_dim: int = 1, **kwargs):
        """
        :param team_count: (int) number of teams
        :param rating_dim: (int) dimensionality of the symbolical rating
        :keyword default: (float) default rating of all teams, default value = 1000.
        :keyword gamma: (float) impact scale of goal difference, default value = 2.
        :keyword c: (float) rating meta parameter, default value = 3.
        :keyword d: (float) rating meta parameter, default value = 500.
        :keyword k: (float) learning rate, default value = 3.
        """
        assert rating_dim > 0 and isinstance(rating_dim, int)

        super(EloSymbolical, self).__init__(team_count, **kwargs)

        self.rating_dim = rating_dim
        if rating_dim > 1:
            default = kwargs.get('default', self._params['default'])
            mean = default
            std = default / 100.

            if True:
                self.elo = nn.Parameter(torch.normal(mean, std, (team_count, rating_dim), dtype=torch.float64))
            else:
                self.elo = nn.Parameter(torch.full((team_count, rating_dim), default, dtype=torch.float64))
            self.ratings = [self.elo]

    def forward(self, matches: Matches):
        home, away = matches

        return _sym_fn(self.elo[home, :], self.elo[away, :], self.c, self.d, self.k)


class _SymFunction(torch.autograd.Function):
    @staticmethod
    def forward(home_rating: Tensor, away_rating: Tensor, c: Tensor, d: Tensor, k: float) -> Tensor:
        assert home_rating.shape == away_rating.shape

        E_H = 1 / (1 + torch.pow(c, ((away_rating - home_rating) / d)))
        return E_H

    @staticmethod
    def setup_context(ctx, inputs, output):
        home_rating, away_rating, c, d, k = inputs
        E_H = output
        ctx.save_for_backward(home_rating, away_rating, c, d, E_H, k)

    @staticmethod
    def backward(ctx, grad_output):
        home_rating, away_rating, c, d, E_H, k = ctx.saved_tensors

        cnst = E_H * (1 - E_H)

        grad_a_rtg = - (torch.log(c) / d) * cnst
        grad_h_rtg = - grad_a_rtg

        rating_diff = away_rating - home_rating

        grad_c = - (rating_diff / (d * c)) * cnst
        grad_d = (rating_diff * torch.log(c) / torch.pow(d, 2)) * cnst

        grad_h_rtg = (grad_output * grad_h_rtg)
        grad_a_rtg = (grad_output * grad_a_rtg)
        grad_c = torch.mean(grad_output * grad_c)
        grad_d = torch.mean(grad_output * grad_d)

        return k * grad_h_rtg, k * grad_a_rtg, grad_c, grad_d, None


# create alias
_sym_fn = _SymFunction.apply
