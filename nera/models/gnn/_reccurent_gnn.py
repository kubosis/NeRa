import torch
import torch.nn as nn

from abc import abstractmethod


class RecurrentGNN(nn.Module):
    def __init__(self, discount: float, debug: bool, correction: bool):

        """
        Basic nn.Module implementation for saving the recurrent GNN model edges and edge weights
        """

        assert 0 < discount < 1

        super(RecurrentGNN, self).__init__()

        self.discount = discount
        self.correction = correction

        self.H_edge_index = None
        self.H_edge_weight = None

        self.debug = debug
        self.embedding_progression = []

    def _copy_index(self, edge_index, edge_weight, h):
        if self.debug:
            # copy hidden state to embedding progression if debugging
            self.embedding_progression.append(h.detach().clone().numpy())

        if edge_weight is None:
            new_edge_weight = torch.ones_like(edge_index[0, :]).detach().to(torch.float)
        else:
            new_edge_weight = edge_weight.detach().clone().to(torch.float)
        if self.H_edge_weight is None:
            self.H_edge_weight = new_edge_weight
        else:
            self.H_edge_weight *= self.discount
            # weight correction
            if self.correction:
                self.H_edge_weight[-2:] = torch.abs(self.H_edge_weight[-2:])
            self.H_edge_weight = torch.cat([self.H_edge_weight, new_edge_weight])

        new_edge_index = edge_index.detach().clone()
        if self.H_edge_index is None:
            self.H_edge_index = new_edge_index
        else:
            self.H_edge_index = torch.cat([self.H_edge_index, new_edge_index], dim=1)

    def reset_index(self):
        self.H_edge_index = None
        self.H_edge_weight = None

    @abstractmethod
    def forward(self, edge_index, home, away, edge_weight=None):
        pass
