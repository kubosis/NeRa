import torch
import torch.nn as nn

from .._general_rating import _GeneralRating
from .._general_rating import *


class BerrarModel(_GeneralRating):
    _params = {
        'alpha_h': torch.tensor(180, dtype=torch.float64),
        'alpha_a': torch.tensor(180, dtype=torch.float64),
        'beta_h': torch.tensor(180, dtype=torch.float64),
        'beta_a': torch.tensor(180, dtype=torch.float64),
        'bias_h': torch.tensor(0, dtype=torch.float64),
        'bias_a': torch.tensor(0, dtype=torch.float64),
        'lr_h_att': torch.tensor(0.1, dtype=torch.float64),
        'lr_a_att': torch.tensor(0.1, dtype=torch.float64),
        'lr_h_def': torch.tensor(0.1, dtype=torch.float64),
        'lr_a_def': torch.tensor(0.1, dtype=torch.float64)
    }

    _learnable = {key: False for key in _params}
    _learnable.update({
        'beta_h': True,
        'beta_a': True,
        'bias_h': True,
        'bias_a': True,
    })

    def __init__(self, team_count: int, default: float = 1000., **kwargs):
        super(BerrarModel, self).__init__(self._params, self._learnable, **kwargs)

        self.h_att = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.h_def = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.a_att = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.a_def = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.ratings = [self.h_att, self.h_def, self.a_att, self.a_def]

        self.g_a, self.g_h = None, None
