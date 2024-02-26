import torch.nn as nn
import torch


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weight):
        weight = weight.detach()
        return torch.mean(weight * torch.pow(prediction - target, 2))
