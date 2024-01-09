"""
gnn.py

contains class for GNN model without spatial temporalities

this file is heavily based on https://github.com/perevale/matches_prediction/tree/master/src project
made by Aleksandra Pereverzeva

"""

import torch

from ._model import GeneralGNNModel


class GNNModel(GeneralGNNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, data, home, away):
        edge_index, edge_weight = data.edge_index, data.edge_weight
        if hasattr(self, 'num_teams'):
            num_teams = self.num_teams
        else:
            num_teams = data.n_teams
        x = torch.tensor(list(range(num_teams)))
        x = self.embedding(x).reshape(-1, self.embed_dim)

        if len(edge_weight) > 0:
            x = self.conv_layers[0](x, edge_index, edge_weight)
        else:
            x = self.conv_layers[0](x, edge_index)
        x = self.activation(x)
        x = self.drop(x)

        for i in range(self.n_conv - 1):
            if len(edge_weight) > 0:
                x = self.activation(self.conv_layers[i + 1](x, data.edge_index, edge_weight))
            else:
                x = self.activation(self.conv_layers[i + 1](x, data.edge_index))

        x = torch.cat([x[home], x[away]], dim=1)

        for i in range(self.n_dense):
            x = self.activation(self.lin_layers[i](x))
            x = self.drop(x)

        x = self.out(x)
        return x.reshape(-1, self.target_dim)
