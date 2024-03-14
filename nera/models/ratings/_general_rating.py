from typing import Sequence

import torch.nn as nn
import torch
import numpy as np

Matches = Sequence[np.ndarray]
Result = Sequence[np.ndarray]
Index = int


class _GeneralRating(nn.Module):
    def __init__(self, params: dict, learnable: dict, **kwargs):
        super(_GeneralRating, self).__init__()

        self.hyperparams = []

        self._hp_backup = {}

        self.hp_grad = False  # compute gradient of hyper params?
        if 'hp_grad' in kwargs:
            self.hp_grad = kwargs['hp_grad']

        for elem in params.keys():
            if elem in kwargs:
                par = kwargs[elem]
                if not isinstance(par, torch.Tensor):
                    par = torch.tensor(par, dtype=torch.float64)
            else:
                par = params[elem]

            if self.hp_grad and learnable[elem]:
                par.requires_grad = True
                par = nn.Parameter(par)
                setattr(self, elem, par)
                self.hyperparams.append(getattr(self, elem))
                self._hp_backup[elem] = par
            else:
                par = par.detach()
                setattr(self, elem, par)

        self.home, self.away = None, None

        self.is_rating = True
        self.is_manual = False

        self.type = None

    def reset_hyperparams(self):
        self.hyperparams = []
        for key, val in self._hp_backup.values():
            setattr(self, key, val)
            self.hyperparams.append(val)
