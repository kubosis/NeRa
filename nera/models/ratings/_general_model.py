import torch.nn as nn
import torch


class GeneralRating(nn.Module):
    def __init__(self, params: dict, **kwargs):
        super(GeneralRating, self).__init__()

        self.hp_grad = False  # compute gradient of hyper params?
        if 'hp_grad' in params:
            self.hp_grad = params['hp_grad']

        for elem in params.keys():
            if elem in kwargs:
                par = kwargs[elem]
                if not isinstance(par, torch.Tensor):
                    par = torch.tensor(par, dtype=torch.float64)
            else:
                par = params[elem]

            if self.hp_grad:
                par.requires_grad = True
                par = nn.Parameter(par)
            else:
                par = par.detach()

            setattr(self, elem, par)

        self.home, self.away = None, None
