from torch import Tensor

from ._model import *


class EloFunction(torch.autograd.Function):
    @staticmethod
    def forward(home_rating: Tensor, away_rating: Tensor, c: Tensor, d: Tensor) -> Tensor:
        E_H = 1 / (1 + torch.pow(c, ((away_rating - home_rating) / d)))
        return E_H

    @staticmethod
    def setup_context(ctx, inputs, output):
        home_rating, away_rating, c, d = inputs
        E_H = output
        ctx.save_for_backward(home_rating, away_rating, c, d, E_H)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        home_rating, away_rating, c, d, E_H = ctx.saved_tensors

        cnst = E_H * (1 - E_H)

        grad_a_rtg = - (torch.log(c) / d) * cnst
        grad_h_rtg = - grad_a_rtg

        rating_diff = away_rating - home_rating

        grad_c = - (rating_diff / (d * c)) * cnst
        grad_d = (rating_diff * torch.log(c) / torch.pow(d, 2)) * cnst

        grad_h_rtg = (grad_output * grad_h_rtg)
        grad_a_rtg = (grad_output * grad_a_rtg)
        grad_c = (grad_output * grad_c)
        grad_d = (grad_output * grad_d)

        return grad_h_rtg, grad_a_rtg, grad_c, grad_d


elo_function = EloFunction.apply


class EloGrad(EloModel):
    """
    Elo with gradient with manual backward pass

    Loss function should be weighted MSE. Weight is the goal difference in the match raised to the power of
    EloGrad.gamma parameter
    """

    def __init__(self, team_count: int, cd_grad: bool = False, **kwargs):
        """
        :param team_count: (int) number of teams
        :keyword default: (float) default rating of all teams, default value = 1000.
        :keyword gamma: (float) impact scale of goal difference, default value = 2.
        :keyword c: (1, torch.Tensor, float64) rating meta parameter, default value = 3.
        :keyword d: (1, torch.Tensor, float64) rating meta parameter, default value = 500.
        :keyword k: (float) learning rate, default value = 2.
        """
        super(EloGrad, self).__init__(team_count, cd_grad=cd_grad, **kwargs)

    def forward(self, matches: Matches):
        self.home, self.away = matches
        return elo_function(self.rating[self.home], self.rating[self.away], self.c, self.d)
