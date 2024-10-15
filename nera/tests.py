# scripts for easier manipulation
from datetime import timedelta
from itertools import product

import numpy as np
import pandas as pd
import torch.nn

from data import *
from dummy import get_dummy_df, Dummy
from models.ratings import (
    EloSymbolical,
    Elo,
    Berrar,
    EloManual, BerrarManual
)
from reference import *
from models.gnn import GConvElman, RatingRGNN
from trainer import Trainer


# simple testing and evaluation functions for development ----------------------------------------------
def eval_symbolic_elo(transform):
    dataset = transform.get_dataset(node_f_extract=False, edge_f_one_hot=True)

    team_count = transform.num_teams

    elo_sym = EloSymbolical(team_count=team_count, rating_dim=3, hp_grad=True, k=5)
    trainer = Trainer(dataset)
    trainer.model = elo_sym
    trainer.train_ratio = 0.8
    trainer.set_lr(0.1, "rating")
    acc_sym = trainer.train(epochs=5, val_ratio=0.2, verbose=True)
    trainer.test(verbose=True)

    print(elo_sym.elo[:5])
    print(acc_sym)


def print_embedding_progression(gnn, trainer, team_count, verbose, embed_dim):
    h = torch.tensor(list(range(gnn.team_count)))
    h = gnn.embedding(h).reshape(-1, gnn.embed_dim)
    gnn.embedding_progression.append(h.clone().detach().numpy())
    progression = np.array(gnn.embedding_progression).reshape(
        -1,
        team_count * embed_dim,
    )
    # embedding_progression = pd.DataFrame(np.array(gnn.embedding_progression).reshape(4, -1), columns=['A', 'B', 'C', 'D'])

    delta = np.zeros((progression.shape[0] - 1, progression.shape[1]))
    for i in range(progression.shape[0] - 1):
        delta[i, :] = -progression[i, :] + progression[i + 1, :]

    print(
        f"Embedding progression\n{np.around(progression, decimals=4)}\ndelta scaled 100times:\n{np.around(100 * delta, decimals=4)}\n--------------------"
    )

    if verbose:
        for name, param in trainer.model.named_parameters():
            print(f"{name}: {param}\n")


def recursive_to(model, device):
    model.to(device)  # Move the model itself to the device
    for elem in model.gconv_layers:
        elem.to(device)
    for elem in model.linear_layers:
        elem.to(device)


def test_eval_rating_gnn(transform, **kwargs):
    dataset_str = kwargs.pop("dataset_str")
    if dataset_str != "Premier":
        dataset = transform.get_dataset(
            node_f_extract=False, edge_f_one_hot=True, drop_draws=True
        )
        target_dim = 2
    else:
        dataset = transform.get_dataset(
            node_f_extract=False, edge_f_one_hot=True, drop_draws=False
        )
        target_dim = 3

    embed_dim = 32
    rtg = kwargs.pop("rtg")

    model = RatingRGNN(
        team_count=transform.num_teams,
        embed_dim=embed_dim,
        target_dim=target_dim,
        discount=0.94,
        correction=False,
        activation="lrelu",
        rgnn_conv="GCONV_ELMAN",
        normalization="rw",
        graph_conv="ChebConv",
        dense_layers=1,
        conv_layers=11,
        dropout_rate=0.2,
        rating=rtg,
        K=8,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recursive_to(model, device)

    trainer = Trainer(
        dataset,
        model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        lr=0.03,
        lr_rating=3.72,
        train_ratio=1,
    )
    predictions_arr = []
    trn_acc, val_acc = trainer.train(epochs=1, val_ratio=0, verbose=True, bidir=True, predictions=predictions_arr)

    if rtg is None:
        rtg = "norating"

    np.savetxt(rtg + "_" + dataset_str + "_val.log", val_acc)
    np.savetxt(rtg + "_" + dataset_str + "_train.log", trn_acc)

    np.savetxt("best_predictions.txt", predictions_arr, fmt="%.4f")

    # print_embedding_progression(model, trainer, transform.num_teams, True, embed_dim)


def test_dummy_id_all(
        test_fn=test_eval_rating_gnn,
        dummy_id=0,
        conf_len=3,
        conf_chars="ha",
        verbose=False,
        rating="elo",
):
    for config in product(conf_chars, repeat=conf_len):
        print(f"config: {config}")
        config = "".join(config)
        dummy = get_dummy_df(dummy_id, conf=config)
        transform = DataTransformation(dummy, timedelta(days=365))
        test_fn(transform, rating=rating, verbose=verbose)


def test_dummy_generate_all(
        test_fn=test_eval_rating_gnn,
        match_count=3,
        team_count=4,
        conf_chars="ha",
        verbose=False,
        rating="elo",
):
    for config in product(conf_chars, repeat=match_count):
        print(f"config: {config}")
        config = "".join(config)
        dummy = Dummy.generate_dummy(match_count, team_count, conf=config)
        transform = DataTransformation(dummy, timedelta(days=365))
        test_fn(transform, verbose=verbose, rating=rating)


def test_dummy_id_one(
        test_fn=test_eval_rating_gnn, dummy_id=0, conf="hhh", verbose=True, rating="elo"
):
    dummy = get_dummy_df(dummy_id, conf=conf)
    transform = DataTransformation(dummy, timedelta(days=365))
    test_fn(transform, verbose=verbose, rating=rating)
    print("Config: ", tuple(conf))


# ---------------------------------------------------------------------------------------------------


def eval_manual(df, dataset_str, rating="Elo"):
    transform = DataTransformation(df, timedelta(days=365))
    if dataset_str != "Premier":
        dataset = transform.get_dataset(
            node_f_extract=False, edge_f_one_hot=True, drop_draws=True
        )
    else:
        dataset = transform.get_dataset(
            node_f_extract=False, edge_f_one_hot=True, drop_draws=False
        )

    if rating == "Elo":
        r = EloManual(transform.num_teams)
    elif rating == "Berrar":
        r = BerrarManual(transform.num_teams)
    else:
        ...
        # r = PiManual(transform.num_teams)
    trainer = Trainer(dataset, r, train_ratio=1)
    pred_arr = []
    trn_acc, val_acc = trainer.train(epochs=1, val_ratio=0, verbose=True, bidir=True, predictions=pred_arr)
    np.savetxt("Manual" + rating + "_" + dataset_str + "_val.log", val_acc)
    np.savetxt("Manual" + rating + "_" + dataset_str + "_train.log", trn_acc)
    np.savetxt(f"{rating}_predictions.txt", np.array(pred_arr), fmt="%.4f")


def nbl(rtg):
    da = DataAcquisition()
    df = da.get_data(FROM_CSV, fname="../resources/european_leagues_basketball.csv")
    df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")
    filtered_df = df[df['League'] == 'NBL']
    filtered_df = filtered_df.reset_index()
    filtered_df = filtered_df.sort_values(by='DT', ascending=False)
    transform = DataTransformation(filtered_df, timedelta(days=365))
    if "manual" in rtg:
        rating = rtg.split("_")[0]
        eval_manual(filtered_df, "NBL", rating)
    else:
        test_eval_rating_gnn(transform, rtg=rtg, dataset_str="NBL")


def test_eval(rtg, dataset_str, fname):
    logger.info(f"Testing {rtg} on {dataset_str}")

    da = DataAcquisition()
    df = da.get_data(FROM_CSV, fname=fname)
    df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by='DT', ascending=False)
    transform = DataTransformation(df, timedelta(days=365))

    if "manual" in rtg:
        rating = rtg.split("_")[0]
        eval_manual(df, dataset_str, rating)
    else:
        test_eval_rating_gnn(transform, rtg=rtg, dataset_str=dataset_str)


def main():
    # torch.manual_seed(42)
    # test_dummy_id_one(
    #    test_fn=test_eval_rating_gnn, conf="hah", dummy_id=0
    # )
    # test_dummy_id_all(test_fn=gnn_rating_test, rating='berrar', verbose=True)
    nbl("Elo_manual")
    #nbl("pi")
    #test_eval("pi", "european_basket", "../resources/european_leagues_basketball.csv")
    # test_eval("Berrar_manual", "Premier", "../resources/Premier_League_England.csv")
    # test_eval("elo", "NFL", "../resources/NFL_USA.csv")


if __name__ == "__main__":
    main()
