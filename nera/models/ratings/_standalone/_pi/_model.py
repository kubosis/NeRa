from ..._general_rating import _GeneralRating
from nera.models.ratings._general_rating import *


class PiModel(_GeneralRating):
    _params = {
        "lambda_": torch.tensor(0.1, dtype=torch.float64),
        "gamma": torch.tensor(0.1, dtype=torch.float64),
        "c": torch.tensor(100, dtype=torch.float64),
    }

    _learnable = {key: False for key in _params}
    _learnable.update(
        {
            "lambda_": True,
            "gamma": True,
            "c": True,
        }
    )

    def __init__(self, team_count: int, default: float = 1000.0, **kwargs):
        super(PiModel, self).__init__(self._params, self._learnable, **kwargs)

        self.home_rating = nn.Parameter(
            torch.full((team_count,), default, dtype=torch.float64)
        )
        self.away_rating = nn.Parameter(
            torch.full((team_count,), default, dtype=torch.float64)
        )
        self.ratings = [self.home_rating, self.away_rating]

        self.g_a, self.g_h = None, None

        self.type = "pi"
