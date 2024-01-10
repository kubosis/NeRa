"""
baseline.py

this file is heavily based on https://github.com/perevale/matches_prediction/tree/master/src project
made by Aleksandra Pereverzeva

"""

import torch
from torch.nn import Embedding, Linear, Dropout, ModuleList
from torch.nn import LogSoftmax, ReLU, Tanh, LeakyReLU

activations = {
    'relu': ReLU(),
    'tanh': Tanh(),
    'leaky': LeakyReLU(0.2)
}


class FlatModel(torch.nn.Module):
    def __init__(self, n_teams, out_dim=3, embed_dim=10, pretrained_weights=None, n_dense=4, dense_dims=(4, 4, 4, 32),
                 act_f='leaky', p_dropout=0.1, **kwargs):
        super(FlatModel, self).__init__()
        # set hyperparameters for the model
        self.n_teams = n_teams
        self.out_dim = out_dim
        self.activation = activations[act_f]
        self.n_dense = n_dense

        # set the layers to be used in the model
        if pretrained_weights is not None:
            self.embedding = Embedding.from_pretrained(pretrained_weights)
        else:
            self.embedding = Embedding(n_teams, embed_dim)

        lin_layers = [torch.nn.Linear(embed_dim * 2, dense_dims[0])]
        for i in range(n_dense - 2):
            lin_layers.append(torch.nn.Linear(dense_dims[i], dense_dims[i + 1]))
        lin_layers.append(torch.nn.Linear(dense_dims[n_dense - 2], self.out_dim))
        self.lin_layers = ModuleList(lin_layers)

        self.out = LogSoftmax(dim=1)

        self.drop = Dropout(p=p_dropout)

    def forward(self, data, team_home, team_away):
        home_emb = self.embedding(team_home)
        away_emb = self.embedding(team_away)
        x = torch.cat((home_emb, away_emb), 1)

        for layer in self.lin_layers:
            x = self.activation(layer(x))
            x = self.drop(x)

        x = self.out(x)
        return x.reshape(-1, self.out_dim)
