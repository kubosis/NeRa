from .._general_rating import _GeneralRating
from .._general_rating import *


class BerrarModel(_GeneralRating):
    _params = {
        'alpha_h': torch.tensor(180, dtype=torch.float64),
        'alpha_a': torch.tensor(180, dtype=torch.float64),
        'beta_h': torch.tensor(2, dtype=torch.float64),
        'beta_a': torch.tensor(2, dtype=torch.float64),
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

        self.att_ = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.def_ = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.ratings = [self.att_, self.def_]

        self.g_a, self.g_h = None, None
