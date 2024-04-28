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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.store = True  # true on training, false on validation
        self.last_match_training = False

    def _copy_index(self, edge_index, edge_weight, h):
        if not self.store:
            #self._copy_index_val(edge_index, edge_weight)
            return

        if self.debug:
            # copy hidden state to embedding progression if debugging
            self.embedding_progression.append(h.detach().clone().numpy())

        if edge_weight is None:
            new_edge_weight = torch.ones_like(edge_index[0, :]).detach().to(self.device, dtype=torch.float)
        else:
            new_edge_weight = edge_weight.detach().clone().to(self.device, dtype=torch.float)
        if self.H_edge_weight is None:
            self.H_edge_weight = new_edge_weight
        else:
            self.H_edge_weight *= self.discount
            # weight correction
            if self.correction:
                self.H_edge_weight[-2:] = torch.abs(self.H_edge_weight[-2:])
            self.H_edge_weight = torch.cat([self.H_edge_weight, new_edge_weight]).to(self.device, dtype=torch.float)

        new_edge_index = edge_index.detach().clone().to(self.device)
        if self.H_edge_index is None:
            self.H_edge_index = new_edge_index
        else:
            self.H_edge_index = torch.cat([self.H_edge_index, new_edge_index], dim=1).to(self.device)

        # cutoff
        max_val = new_edge_weight[0].item()
        cutoff_mask = self.H_edge_weight > max_val * 1e-10
        self.H_edge_weight = self.H_edge_weight[cutoff_mask]
        self.H_edge_index = self.H_edge_index[:, cutoff_mask]

    def _copy_index_val(self, edge_index, edge_weight):
        assert self.H_edge_index is not None

        if edge_weight is None:
            new_edge_weight = torch.ones_like(edge_index[0, :]).detach().to(self.device, dtype=torch.float)
        else:
            new_edge_weight = edge_weight.detach().clone().to(self.device, dtype=torch.float)

        if self.last_match_training:
            # last match was training match
            self.H_edge_weight *= self.discount
            if self.correction:
                self.H_edge_weight[-2:] = torch.abs(self.H_edge_weight[-2:])
            self.last_match_training = False
        else:
            # remove last to edge since the last match is validation match
            self.H_edge_weight = self.H_edge_weight[:-2]
            self.H_edge_index = self.H_edge_index[:, :-2]

        new_edge_index = edge_index.detach().clone().to(self.device)
        self.H_edge_index = torch.cat([self.H_edge_index, new_edge_index], dim=1).to(self.device)
        self.H_edge_weight = torch.cat([self.H_edge_weight, new_edge_weight]).to(self.device, dtype=torch.float)

    def reset_index(self):
        self.H_edge_index = None
        self.H_edge_weight = None

    def store_index(self, store: bool):
        self.store = store
        if store and self.H_edge_index is not None:
            # last edge was validation. we have to delete it
            # we dont reset discount on edges since there is supposedly long delay between last training edge and new
            # training edges -> last edges will be discounted two times instead
            self.H_edge_index = self.H_edge_index[:, :-2]
            self.H_edge_weight = self.H_edge_weight[:-2]

        if not store:
            self.last_match_training = True

    @abstractmethod
    def forward(self, edge_index, home, away, edge_weight=None):
        pass
