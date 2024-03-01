from typing import Sequence

import torch
import torch.nn as nn
import numpy as np

from .._general_rating import _GeneralRating
from .._general_rating import *


class EloModel(_GeneralRating):
    _params = {
        'k': torch.tensor(3., dtype=torch.float64),
        'gamma': torch.tensor(2., dtype=torch.float64),
        'c': torch.tensor(3., dtype=torch.float64),
        'd': torch.tensor(500., dtype=torch.float64),
    }

    _learnable = {key: False for key in _params}
    _learnable.update({
        'c': True,
        'd': True,
    })

    def __init__(self, team_count: int, default: float = 1000., **kwargs):
        super(EloModel, self).__init__(self._params, self._learnable, **kwargs)

        self.elo = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.ratings = [self.elo]

        self.E_H = None
