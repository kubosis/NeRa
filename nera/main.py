# scripts for easier manipulation
import torch
from datetime import timedelta
from itertools import product

import numpy as np
import torch_geometric.nn.aggr as aggr

from nera.data import *
from nera.dummy import get_dummy_df, Dummy
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
    trainer = Trainer(dataset, gnn, loss_fn=torch.nn.CrossEntropyLoss, lr=0.1)
    trainer.train_ratio = 1
    acc_sym = trainer.train(epochs=500, val_ratio=0, verbose=True)
    trainer.test(verbose=True)


def simple_gnn_test(transform, verbose=False):
    dataset = transform.get_dataset(node_f_extract=False, edge_f_one_hot=True, drop_draws=True)
    team_count = transform.num_teams

    gnn = RGNN(team_count=team_count, in_channels=1,
               conv_out_channels=1, debug=True, K=2,
               out_channels=2, init_ones_=True, bias=True, normalization='sym',
               graph_conv='GCONV_ELMAN', discount=0.8, aggr='add')
    trainer = Trainer(dataset, gnn, loss_fn=torch.nn.MSELoss(), lr=0.01, lr_rating=1)
    trainer.train_ratio = 1
    _ = trainer.train(epochs=1, val_ratio=0, verbose=verbose, bidir=True)
    h = torch.tensor(list(range(gnn.team_count)))
    h = gnn.embedding[h].reshape(-1, gnn.embed_dim)
    gnn.embedding_progression.append(h.clone().detach().numpy())
    progression = np.array(gnn.embedding_progression).reshape(-1, team_count)
    #embedding_progression = pd.DataFrame(np.array(gnn.embedding_progression).reshape(4, -1), columns=['A', 'B', 'C', 'D'])

    delta = np.zeros((progression.shape[0] - 1, progression.shape[1]))
    for i in range(progression.shape[0] - 1):
        delta[i, :] = - progression[i, :] + progression[i + 1, :]

    print(
        f'Embedding progression\n{progression}\ndelta scaled 100times:\n{np.around(100*delta, decimals=4)}\n--------------------')

    if verbose:
        for name, param in trainer.model.named_parameters():
            print(f"{name}: {param}\n")


def test_dummy_id_all(dummy_id=0, conf_len=3, conf_chars='ha', verbose=False):
    for config in product(conf_chars, repeat=conf_len):
        print(f'config: {config}')
        config = ''.join(config)
        dummy = get_dummy_df(dummy_id, conf=config)
        transform = DataTransformation(dummy, timedelta(days=365))
        simple_gnn_test(transform, verbose=verbose)


def test_dummy_generate_all(match_count=3, team_count=4, conf_chars='ha', verbose=False):
    for config in product(conf_chars, repeat=match_count):
        print(f'config: {config}')
        config = ''.join(config)
        dummy = Dummy.generate_dummy(match_count, team_count, conf=config)
        transform = DataTransformation(dummy, timedelta(days=365))
        simple_gnn_test(transform, verbose=verbose)


def test_dummy_id_one(dummy_id=0, conf='hhh', verbose=True):
    dummy = get_dummy_df(dummy_id, conf=conf)
    transform = DataTransformation(dummy, timedelta(days=365))
    simple_gnn_test(transform, verbose=verbose)
    print("Config: ", tuple(conf))


def main():
    #torch.manual_seed(42)
    #test_dummy_id_one(verbose=False, conf='hhh', dummy_id=0)
    test_dummy_id_all(verbose=False)


if __name__ == '__main__':
    main()
