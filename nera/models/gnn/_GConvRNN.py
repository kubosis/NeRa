import torch
from torch_geometric.nn import GraphConv


class GConvRNN(torch.nn.Module):
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
            K: int,
            aggr: str = "add",
            bias: bool = True,
    ):
        super(GConvRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.aggr = aggr
        self.bias = bias
        self.hts1 = None
        self._create_conv_layers()

    def _create_conv_layers(self):
        self.conv_whx_x = GraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            bias=self.bias,
            aggr=self.aggr
        )

        self.conv_whh_hs1 = GraphConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            bias=self.bias,
            aggr=self.aggr
        )

        self.conv_wy_y = GraphConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            bias=self.bias,
            aggr=self.aggr
        )

    def _set_hidden_state(self, X):
        if self.hts1 is None:
            return torch.zeros(X.shape[0], self.out_channels).to(X.device).float()
        return self.hts1

    def _calculate_ht(self, X, edge_index, edge_weight, H):
        Whx_g_xt = self.conv_whx_x(X, edge_index, edge_weight)
        Whh_g_hts1 = self.conv_whh_hs1(H, edge_index, edge_weight)
        ht = torch.sigmoid(Whx_g_xt + Whh_g_hts1)
        return ht

    def _calculate_yt(self, ht, edge_index, edge_weight):
        Wy_g_ht = self.conv_wy_y(ht, edge_index, edge_weight)
        yt = torch.sigmoid(Wy_g_ht)
        return yt

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
        hts1 = self._set_hidden_state(X)
        ht = self._calculate_ht(X, edge_index, edge_weight, hts1)
        yt = self._calculate_yt(ht, edge_index, edge_weight)
        return yt
