from typing import Sequence

import torch
import torch.nn as nn
import numpy as np

from .._general_model import GeneralRating

Matches = Sequence[np.ndarray]
Result = Sequence[np.ndarray]
Index = int


class EloModel(GeneralRating):
    _params = {
        'k': torch.tensor(3., dtype=torch.float64),
        'gamma': torch.tensor(2., dtype=torch.float64),
        'c': torch.tensor(3., dtype=torch.float64),
        'd': torch.tensor(500., dtype=torch.float64),
    }

    def __init__(self, team_count: int, default: float = 1000., **kwargs):
        super(EloModel, self).__init__(self._params, **kwargs)

        self.rating = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.E_H = None

