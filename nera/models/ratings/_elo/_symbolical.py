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
            self.elo = nn.Parameter(torch.full((team_count, rating_dim), default, dtype=torch.float64))

    def forward(self, matches: Matches):
        pass


class _SymFunction(torch.autograd.Function):
    @staticmethod
    def forward(home_rating: Tensor, away_rating: Tensor, c: Tensor, d: Tensor) -> Tensor:
        pass

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        pass


# create alias
_sym_fn = _SymFunction.apply
