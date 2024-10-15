from ..._general_rating import _GeneralRating
from ..._general_rating import *


class EloModel(_GeneralRating):
    _params = {
        "k": torch.tensor(3.0, dtype=torch.float64),
        "gamma": torch.tensor(0.2, dtype=torch.float64),
        "c": torch.tensor(3.0, dtype=torch.float64),
        "d": torch.tensor(8, dtype=torch.float64),
        "default": torch.tensor(1.0, dtype=torch.float64),
    }

    _learnable = {key: False for key in _params}
    _learnable.update(
        {
            "c": True,
            "d": True,
        }
    )

    def __init__(self, team_count: int, **kwargs):
        assert isinstance(team_count, int)

        super(EloModel, self).__init__(self._params, self._learnable, **kwargs)

        default = kwargs.get("default", self._params["default"])

        self.elo = nn.Parameter(torch.full((team_count,), default, dtype=torch.float64))
        self.ratings = [self.elo]

        self.E_H = None

        self.type = "elo"
