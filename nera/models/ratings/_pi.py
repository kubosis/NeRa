import torch.nn as nn
import torch


class Pi(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, home, away):
        ...
