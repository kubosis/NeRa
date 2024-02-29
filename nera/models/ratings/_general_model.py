import torch.nn as nn


class GeneralRating(nn.Module):
    def __init__(self, params: dict, **kwargs):
        super(GeneralRating, self).__init__()
        for elem in params.keys():
            if elem in kwargs:
                setattr(self, elem, kwargs[elem])
            else:
                setattr(self, elem, params[elem])
        self.home, self.away = None, None
