# scripts for easier manipulation

import torch
import pandas as pd
from datetime import timedelta
from datetime import datetime

import numpy as np

from nera.data import *
from nera.models.ratings import EloAnalytical, EloManual, EloNumerical, EloSymbolical
from nera.reference import *
from nera.trainer import Trainer
from nera.utils import print_rating_diff


def eval_symbolic_elo():
    da = DataAcquisition()
    df = da.get_data(FROM_CSV, fname="../resources/other_leagues.csv")
    df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")
    # df = df[(df['League'] != 'EuroLeague') & (df['League'] != 'EuroCup')]
    df = df.reset_index()
    df = df.sort_values(by='DT', ascending=False)

    transform = DataTransformation(df, timedelta(365))
    dataset = transform.get_dataset(node_f_extract=False, edge_f_one_hot=True)

    team_count = transform.num_teams

    elo_sym = EloSymbolical(team_count=transform.num_teams, rating_dim=3, hp_grad=True, k=5)
    trainer = Trainer(dataset)
    trainer.model = elo_sym
    trainer.train_ratio = 0.8
    trainer.set_lr(0.1, 'rating')
    acc_sym = trainer.train(epochs=5, val_ratio=0.2, verbose=True)
    trainer.test(verbose=True)

    print(elo_sym.elo[:5])

def main():
    eval_symbolic_elo()

if __name__ == '__main__':
    main()
