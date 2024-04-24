import torch

_params = {
    "alpha_h": torch.tensor(180, dtype=torch.float64, requires_grad=True),
    "alpha_a": torch.tensor(180, dtype=torch.float64, requires_grad=True),
    "beta_h": torch.tensor(2, dtype=torch.float64, requires_grad=True),
    "beta_a": torch.tensor(2, dtype=torch.float64, requires_grad=True),
    "bias_h": torch.tensor(0, dtype=torch.float64, requires_grad=True),
    "bias_a": torch.tensor(0, dtype=torch.float64, requires_grad=True),
    "lr_h_att": torch.tensor(0.1, dtype=torch.float64, requires_grad=True),
    "lr_a_att": torch.tensor(0.1, dtype=torch.float64, requires_grad=True),
    "lr_h_def": torch.tensor(0.1, dtype=torch.float64, requires_grad=True),
    "lr_a_def": torch.tensor(0.1, dtype=torch.float64, requires_grad=True),
}

_learnable = {key: False for key in _params}
_learnable.update(
    {
        "beta_h": True,
        "beta_a": True,
        "bias_h": True,
        "bias_a": True,
        "alpha_h": True,
        "alpha_a": True,
    }
)
