import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from torch_geometric_temporal.signal import DynamicGraphStaticSignal
from typing import Sequence


class DataTransformation:
    def __init__(self, df: pd.DataFrame, snapshot_duration: timedelta):
        """ Class Data Transformation
            Transforms tabular matches dataset into dataset used by PyTorch Geometric Temporal

            :param df: Dataframe to transform
            :param snapshot_duration: time duration of each graph snapshot
        """
        needed_columns = ['league', 'DT', 'Home', 'Away', 'Winner', 'Home_points', 'Away_points']
        self.df: pd.DataFrame = df.loc[:, df.columns.intersection(needed_columns)]
        self.team_mapping: dict = {}
        self.wld_mapping: dict = {}
        self.delta: timedelta = snapshot_duration
        self.teams: list[str] = []
        self.num_teams = 0

    def _create_teams_mapping(self) -> None:
        teams = pd.concat([self.df['Home'], self.df['Away']], axis=0).unique()
        team_ids = list(range(len(teams)))
        self.team_mapping = dict(zip(teams, team_ids))
        self.teams = list(teams)
        self.num_teams = len(teams)

        self.df['Away'] = self.df['Away'].map(self.team_mapping)
        self.df['Home'] = self.df['Home'].map(self.team_mapping)

    def _map_match_outcomes(self, verbose: bool) -> None:
        self.df.loc[self.df['Winner'] == 'home', 'Winner'] = 0
        self.df.loc[self.df['Winner'] == 'away', 'Winner'] = 2
        self.df.loc[self.df['Winner'] == 'draw', 'Winner'] = 1
        if verbose:
            logger.info(f"There are {len(self.df.loc[self.df['Winner'] == 0])} home wins, "
                        f"{len(self.df.loc[self.df['Winner'] == 1])} draws and "
                        f"{len(self.df.loc[self.df['Winner'] == 2])} away wins in the dataset")

    def _extract_node_features(self, use_draws: bool = False) -> Sequence[np.ndarray]:
        delta = self.delta
        start_date = min(self.df['DT'])
        end_date = max(self.df['DT'])

        node_features = []

        while start_date <= end_date:
            df_i = self.df[((start_date + delta >= self.df['DT']) & (start_date <= self.df['DT']))]

            home_wins = np.zeros((self.num_teams,))
            home_losses = np.zeros((self.num_teams,))
            away_wins = np.zeros((self.num_teams,))
            away_losses = np.zeros((self.num_teams,))

            # extract features
            home_wins_series = df_i.loc[df_i['Winner'] == 0].groupby('Home').count()['Winner']
            home_wins[home_wins_series.index] = home_wins_series.values
            home_losses_series = df_i.loc[df_i['Winner'] == 2].groupby('Home').count()['Winner']
            home_losses[home_losses_series.index] = home_losses_series.values
            away_wins_series = df_i.loc[df_i['Winner'] == 2].groupby('Away').count()['Winner']
            away_wins[away_wins_series.index] = away_wins_series.values
            away_losses_series = df_i.loc[df_i['Winner'] == 0].groupby('Away').count()['Winner']
            away_losses[away_losses_series.index] = away_losses_series.values

            # normalize features
            home_wins = (home_wins / np.min(home_wins)) / (np.max(home_wins) - np.min(home_wins))
            home_losses = (home_losses / np.min(home_losses)) / (np.max(home_losses) - np.min(home_losses))
            away_wins = (away_wins / np.min(away_wins)) / (np.max(away_wins) - np.min(away_wins))
            away_losses = (away_losses / np.min(away_losses)) / (np.max(away_losses) - np.min(away_losses))

            if use_draws:
                away_draws = np.zeros((self.num_teams,))
                home_draws = np.zeros((self.num_teams,))
                home_draws_series = df_i.loc[df_i['Winner'] == 1].groupby('Home').count()['Winner']
                home_draws[home_draws_series.index] = home_draws_series.values
                away_draws_series = df_i.loc[df_i['Winner'] == 1].groupby('Away').count()['Winner']
                away_draws[away_draws_series.index] = away_draws_series.values
                home_draws = (home_draws / np.min(home_draws)) / (np.max(home_draws) - np.min(home_draws))
                away_draws = (away_draws / np.min(away_draws)) / (np.max(away_draws) - np.min(away_draws))

                features_i = np.stack([home_wins, home_draws, home_losses,
                                       away_wins, away_draws, away_losses]).transpose()
            else:
                features_i = np.stack([home_wins, home_losses, away_wins, away_losses]).transpose()

            node_features.append(features_i)
            start_date = end_date + delta

        return node_features

    def transform(self, verbose: bool = False, node_features: bool = False):
        """ transform dataframe into dataset used by PyTorch Geometric
            after transformation, mapping of the teams to unique ids is saved to self.mapping
            the transformation is destructive for the original dataframe

            :keyword verbose: print log to console
        """

        # 1)
        self._create_teams_mapping()

        # 2) (win home = 0; win away = 2; draw = 1)
        self._map_match_outcomes(verbose)
