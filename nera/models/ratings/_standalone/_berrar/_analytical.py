from torch import Tensor

from ._model import *


class BerrarAnalytical(BerrarModel):
    """
    Berrar with gradient with analytical backward pass

    Loss function should be MSE, forward pass outputs predicted
    number of goals scored by [home, away] team in Tensor
    """

    def __init__(self, team_count: int, **kwargs):
        """
        :param team_count: number of teams
        :keyword alpha_h: expected number of goals by home team, default = 180
        :keyword beta_h: steepness of exponential for home team, default = 2
        :keyword bias_h:  bias of home team, default = 0
        :keyword alpha_a: expected number of goals by away team, default = 180
        :keyword beta_a: steepness of exponential for away team, default = 2
        :keyword bias_a: bias of away team, default = 0
        """
        super(BerrarAnalytical, self).__init__(team_count, **kwargs)

    def forward(self, matches: Matches):
        h, a = self.home, self.away = matches

        hatt, hdef = self.att_[h], self.def_[h]
        aatt, adef = self.att_[a], self.def_[a]

        ah, bh, yh = self.alpha_h, self.beta_h, self.bias_h
        aa, ba, ya = self.alpha_a, self.beta_a, self.bias_a

        return _berrar_fn(hatt, hdef, aatt, adef, ah, aa, bh, ba, yh, ya)


class _BerrarFunction(torch.autograd.Function):
    @staticmethod
    def forward(hatt, hdef, aatt, adef, ah, aa, bh, ba, yh, ya) -> Tensor:
        ghat_h = ah / (1 + torch.exp(-bh * (hatt + adef) - yh))
        ghat_a = aa / (1 + torch.exp(-ba * (aatt + hdef) - ya))

        ghat_h = ghat_h.unsqueeze(0).unsqueeze(1)
        ghat_a = ghat_a.unsqueeze(0).unsqueeze(1)

        return torch.cat((ghat_h, ghat_a), dim=0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        hatt, hdef, aatt, adef, ah, aa, bh, ba, yh, ya = inputs
        ghat_h = output[0].unsqueeze(0)
        ghat_a = output[1].unsqueeze(0)
        ctx.save_for_backward(
            hatt, hdef, aatt, adef, ah, aa, bh, ba, yh, ya, ghat_h, ghat_a
        )

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        (
            hatt,
            hdef,
            aatt,
            adef,
            ah,
            aa,
            bh,
            ba,
            yh,
            ya,
            ghat_h,
            ghat_a,
        ) = ctx.saved_tensors

        exp_h = torch.exp(bh * (hatt + adef) + yh)
        exp_a = torch.exp(ba * (aatt + hdef) + ya)

        d_ghat_h_d_hatt = ghat_h * bh / (1 + exp_h)
        d_ghat_h_d_adef = ghat_h * bh / (1 + exp_h)

        d_ghat_a_d_aatt = ghat_a * ba / (1 + exp_a)
        d_ghat_a_d_hdef = ghat_a * ba / (1 + exp_a)

        d_loss_d_ah = None
        d_loss_d_aa = None
        d_loss_d_bh = None
        d_loss_d_ba = None
        d_loss_d_yh = None
        d_loss_d_ya = None

        if ctx.needs_input_grad[4]:
            d_ghat_h_d_ah = 1 / (1 + torch.exp(-bh * (hatt + adef) - yh))
            d_loss_d_ah = grad_output[0] * d_ghat_h_d_ah
        if ctx.needs_input_grad[5]:
            d_ghat_a_d_aa = 1 / (1 + torch.exp(-ba * (aatt + hdef) - ya))
            d_loss_d_aa = grad_output[1] * d_ghat_a_d_aa
        if ctx.needs_input_grad[6]:
            d_ghat_h_d_bh = ghat_h * (hatt + adef) / (1 + exp_h)
            d_loss_d_bh = grad_output[0] * d_ghat_h_d_bh
        if ctx.needs_input_grad[7]:
            d_ghat_a_d_ba = ghat_a * (aatt + hdef) / (1 + exp_a)
            d_loss_d_ba = grad_output[1] * d_ghat_a_d_ba
        if ctx.needs_input_grad[8]:
            d_ghat_h_d_yh = ghat_h / (1 + exp_h)
            d_loss_d_yh = grad_output[0] * d_ghat_h_d_yh
        if ctx.needs_input_grad[9]:
            d_ghat_a_d_ya = ghat_a / (1 + exp_a)
            d_loss_d_ya = grad_output[1] * d_ghat_a_d_ya

        d_loss_d_hatt = grad_output[0] * d_ghat_h_d_hatt
        d_loss_d_adef = grad_output[0] * d_ghat_h_d_adef

        d_loss_d_aatt = grad_output[1] * d_ghat_a_d_aatt
        d_loss_d_hdef = grad_output[1] * d_ghat_a_d_hdef

        return (
            d_loss_d_hatt,
            d_loss_d_hdef,
            d_loss_d_aatt,
            d_loss_d_adef,
            d_loss_d_ah,
            d_loss_d_aa,
            d_loss_d_bh,
            d_loss_d_ba,
            d_loss_d_yh,
            d_loss_d_ya,
        )


# create alias
_berrar_fn = _BerrarFunction.apply
