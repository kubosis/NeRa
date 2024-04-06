# scripts for easier manipulation
import torch
import pandas as pd
from datetime import timedelta
from datetime import datetime

import numpy as np

from nera.data import *
from nera.models.ratings import EloAnalytical, EloManual, EloNumerical, EloSymbolical
from nera.reference import *
from nera.models.gnn import GCONVCheb, RGNN
from nera.trainer import Trainer
from nera.utils import print_rating_diff, generate_random_matches


def eval_symbolic_elo(transform):
    dataset = transform.get_dataset(node_f_extract=False, edge_f_one_hot=True)

    team_count = transform.num_teams

    elo_sym = EloSymbolical(team_count=team_count, rating_dim=3, hp_grad=True, k=5)
    trainer = Trainer(dataset)
    trainer.model = elo_sym
    trainer.train_ratio = 0.8
    trainer.set_lr(0.1, 'rating')
    acc_sym = trainer.train(epochs=5, val_ratio=0.2, verbose=True)
    trainer.test(verbose=True)

    print(elo_sym.elo[:5])
    print(acc_sym)

def eval_gnn(transform):
    dataset = transform.get_dataset(node_f_extract=False, edge_f_one_hot=True)
    team_count = transform.num_teams

    gnn = GCONVCheb(team_count=team_count, embed_dim=100)
    trainer = Trainer(dataset, gnn, loss_fn=torch.nn.CrossEntropyLoss)
    trainer.train_ratio = 1
    acc_sym = trainer.train(epochs=500, val_ratio=0, verbose=True)
    trainer.test(verbose=True)


def simple_gnn_test(transform):
    dataset = transform.get_dataset(node_f_extract=False, edge_f_one_hot=True, drop_draws=True)
    team_count = transform.num_teams

    gnn = RGNN(team_count=team_count, embed_dim=1, conv_out_channels=8, debug=True, K=1, target_dim=2)
    trainer = Trainer(dataset, gnn, loss_fn=torch.nn.CrossEntropyLoss, lr=0.01)
    trainer.train_ratio = 1
    _ = trainer.train(epochs=1, val_ratio=0, verbose=True)
    h = torch.tensor(list(range(gnn.team_count)))
    h = gnn.embedding[h].reshape(-1, gnn.embed_dim)
    gnn.embedding_progression.append(h.clone().detach().numpy())
    #embedding_progression = pd.DataFrame(np.array(gnn.embedding_progression).reshape(4, -1), columns=['A', 'B', 'C', 'D'])
    print(f'Embedding progression\n{np.array(gnn.embedding_progression).reshape(-1, 4)}\n--------------------')


def _dummy0():
    delta = timedelta(days=1)
    now = datetime.now()
    data = pd.DataFrame({'DT': [*[now + j * delta for j in range(3)]],
                         'Home': ['A', 'B', 'D'],
                         'Away': ['B', 'C', 'C'],
                         'Winner': ['home', 'home', 'away',],
                         'Home_points': [10, 11, 20],
                         'Away_points': [4, 10, 17,],
                         'League': [*(3 * ['liga'])],
                         })
    return data

def _dummy1():
    delta = timedelta(days=1)
    now = datetime.now()
    data = pd.DataFrame({'DT': [*[now + j * delta for j in range(4)]],
                         'Home': ['A', 'B', 'D', 'D'],
                         'Away': ['B', 'C', 'C', 'A'],
                         'Winner': ['home', 'home', 'away', 'home'],
                         'Home_points': [10, 11, 20, 20],
                         'Away_points': [4, 10, 17, 15],
                         'League': [*(4 * ['liga'])],
                         })
    return data

def get_dummy_df(id=0):
    # dummy dataset
    if id == 0:
        return _dummy0()
    elif id == 1:
        return _dummy1()

def main():
    torch.manual_seed(42)
    dummy = get_dummy_df(0)
    transform = DataTransformation(dummy, timedelta(days=365))
    simple_gnn_test(transform)


if __name__ == '__main__':
    main()
