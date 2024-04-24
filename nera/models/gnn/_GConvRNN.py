import copy

import torch
from torch import nn
from torch_geometric.nn import GraphConv


class GConvElman(torch.nn.Module):
    r"""Our implementation of simple recurrent elman network for graph convolutional neural networks

    :math:`h_{t} = \sigma(W_{hx} *_g x_t + W_{hh} *_g h_{t-1} + b_h)`

    :math:`y_{t} = \sigma(W_{y} *_g h_t + b_y)`

    where :math:`*_g` represents graph convolution operator


    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int = -1,
            aggr: str = "add",
            bias: bool = True,
            init_ones_: bool = True,
    ):
        super(GConvElman, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels > 0 else out_channels
        self.aggr = aggr
        self.bias = bias
        self.first_call = True
        self.H = None
        self._create_conv_layers(init_ones_=init_ones_)

    def _create_conv_layers(self, init_ones_):
        self.conv_whx_x = GraphConv(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            bias=self.bias,
            aggr=self.aggr,
            #flow="source_to_target",
        )

        self.conv_whh_hs1 = GraphConv(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            bias=self.bias,
            aggr=self.aggr,
            #flow="source_to_target",
        )

        self.conv_wy_y = GraphConv(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            bias=self.bias,
            aggr=self.aggr,
            #flow="source_to_target",
        )

        if init_ones_:
            nn.init.eye_(self.conv_whx_x.lin_rel.weight)
            nn.init.eye_(self.conv_whx_x.lin_root.weight)
            nn.init.eye_(self.conv_whh_hs1.lin_rel.weight)
            nn.init.eye_(self.conv_whh_hs1.lin_root.weight)
            nn.init.eye_(self.conv_wy_y.lin_rel.weight)
            nn.init.eye_(self.conv_wy_y.lin_root.weight)

            if self.conv_whx_x.lin_rel.bias is not None:
                nn.init.zeros_(self.conv_whx_x.lin_rel.bias)
            if self.conv_whx_x.lin_root.bias is not None:
                nn.init.zeros_(self.conv_whx_x.lin_root.bias)

            if self.conv_whh_hs1.lin_rel.bias is not None:
                nn.init.zeros_(self.conv_whh_hs1.lin_rel.bias)
            if self.conv_whh_hs1.lin_root.bias is not None:
                nn.init.zeros_(self.conv_whh_hs1.lin_root.bias)

            if self.conv_wy_y.lin_rel.bias is not None:
                nn.init.zeros_(self.conv_wy_y.lin_rel.bias)
            if self.conv_wy_y.lin_root.bias is not None:
                nn.init.zeros_(self.conv_wy_y.lin_root.bias)

    def _set_hidden_state(self, X):
        if self.H is None:
            return torch.zeros(X.shape[0], self.out_channels).to(X.device).float()
        return self.H

    def _calculate_ht(self, X, edge_index, edge_weight, H):
        Whx_g_xt = self.conv_whx_x(X, edge_index, edge_weight)
        Whh_g_hts1 = self.conv_whh_hs1(H, edge_index, edge_weight)
        H = torch.sigmoid(Whx_g_xt + Whh_g_hts1)
        return H

    def _calculate_yt(self, H, edge_index, edge_weight):
        Wy_g_ht = self.conv_wy_y(H, edge_index, edge_weight)
        yt = torch.sigmoid(Wy_g_ht)
        return yt

    def _copy_hidden(self, ht):
        self.H = ht.detach().clone()
        self.H.requires_grad_(True)

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.

        Return types:
            * **y** *(PyTorch Float Tensor)* - Output matrix.
        """
        H = self._set_hidden_state(X)

        H = self._calculate_ht(X, edge_index, edge_weight, H)

        # copy hidden state for next iter
        self._copy_hidden(H)

        yt = self._calculate_yt(H, edge_index, edge_weight)

        return yt
