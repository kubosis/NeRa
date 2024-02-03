import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


def normalize_array(array: np.ndarray) -> np.ndarray:
    denominator = np.max(array) - np.min(array)
    denominator = 1 if denominator == 0 else denominator
    normalized = (array - np.min(array)) / denominator
    return normalized


def one_hot_encode(i: int, max_n: int) -> np.ndarray:
    out = np.zeros((max_n,))
    out[i] = 1
    return out


class DataTransformation:
    def __init__(self, df: pd.DataFrame, snapshot_duration: timedelta):
        """ Class Data Transformation
            Transforms tabular matches dataset into dataset used by PyTorch Geometric Temporal

            :param df: Dataframe to transform
            :param snapshot_duration: time duration of each graph snapshot
        """
        needed_columns = ['League', 'DT', 'Home', 'Away', 'Winner', 'Home_points', 'Away_points']
        self.df: pd.DataFrame = df.loc[:, df.columns.intersection(needed_columns)]
        self.team_mapping: dict = {}
        self.wld_mapping: dict = {}
        self.delta: timedelta = snapshot_duration
        self.teams: list[str] = []
        self.num_teams = 0

        self.start_date = min(self.df['DT'])
        self.end_date = max(self.df['DT'])
        self.snapshot_count = int(np.ceil((self.end_date - self.start_date) / self.delta))

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

    def _extract_node_features(self, discount_factor: float, use_draws: bool, verbose: bool) -> list[np.ndarray]:
        delta = self.delta
        start_date = self.start_date
        end_date = self.end_date

        node_features = []

        home_wins = np.zeros((self.num_teams,))
        home_losses = np.zeros((self.num_teams,))
        away_wins = np.zeros((self.num_teams,))
        away_losses = np.zeros((self.num_teams,))
        away_draws = np.zeros((self.num_teams,))
        home_draws = np.zeros((self.num_teams,))

        i = 0

        while start_date <= end_date:
            df_i = self.df[((start_date + delta >= self.df['DT']) & (start_date <= self.df['DT']))]

            # discount features from t-1 snapshot
            home_wins *= discount_factor
            home_losses *= discount_factor
            away_wins *= discount_factor
            away_losses *= discount_factor
            away_draws *= discount_factor
            home_draws *= discount_factor

            # extract features
            home_wins_series = df_i.loc[df_i['Winner'] == 0].groupby('Home').count()['Winner']
            home_wins[home_wins_series.index] += home_wins_series.values
            home_losses_series = df_i.loc[df_i['Winner'] == 2].groupby('Home').count()['Winner']
            home_losses[home_losses_series.index] += home_losses_series.values
            away_wins_series = df_i.loc[df_i['Winner'] == 2].groupby('Away').count()['Winner']
            away_wins[away_wins_series.index] += away_wins_series.values
            away_losses_series = df_i.loc[df_i['Winner'] == 0].groupby('Away').count()['Winner']
            away_losses[away_losses_series.index] += away_losses_series.values

            # normalize features
            home_wins_norm = normalize_array(home_wins)
            home_losses_norm = normalize_array(home_losses)
            away_wins_norm = normalize_array(away_wins)
            away_losses_norm = normalize_array(away_losses)

            if use_draws:
                home_draws_series = df_i.loc[df_i['Winner'] == 1].groupby('Home').count()['Winner']
                home_draws[home_draws_series.index] = home_draws_series.values
                away_draws_series = df_i.loc[df_i['Winner'] == 1].groupby('Away').count()['Winner']
                away_draws[away_draws_series.index] = away_draws_series.values
                home_draws_norm = normalize_array(home_draws)
                away_draws_norm = normalize_array(away_draws)

                features_i = np.stack([home_wins_norm, home_draws_norm, home_losses_norm,
                                       away_wins_norm, away_draws_norm, away_losses_norm]).transpose()
                node_features.append(features_i)
            else:
                features_i = np.stack([home_wins_norm, home_losses_norm,
                                       away_wins_norm, away_losses_norm]).transpose()
                node_features.append(features_i)

            if verbose:
                logger.info(
                    f"Snapshot {i} of {len(df_i)} matches with the following first 3 features:\n{node_features[i][:3]}")

            start_date += delta
            i += 1

        return node_features

    def _extract_dynamic_edges_and_labels(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """ extract dynamically changing edges and their features """
        delta = self.delta
        start_date = self.start_date
        end_date = self.end_date

        edges = []
        edge_features = []
        labels = []

        leagues = self.df['League'].unique()

        while start_date <= end_date:
            df_i = self.df[((start_date + delta >= self.df['DT']) & (start_date <= self.df['DT']))]

            edges.append(df_i.loc[:, ['Home', 'Away']].to_numpy().T)

            match_outcomes = df_i.loc[:, ['Winner']].values
            match_outcomes = np.apply_along_axis(lambda x: one_hot_encode(x[0], 3), 1, match_outcomes)
            edge_features.append(match_outcomes)

            teams_by_leagues = df_i.groupby('League')

            team_points = np.zeros((self.num_teams,))
            team_ranks = np.zeros((self.num_teams,))
            away_team_wins = df_i.loc[df_i['Winner'] == 2].groupby(['League', 'Away']).count()
            home_team_wins = df_i.loc[df_i['Winner'] == 0].groupby(['League', 'Home']).count()
            team_points[away_team_wins.index.get_level_values(1).to_numpy()] += away_team_wins['Winner'].values
            team_points[home_team_wins.index.get_level_values(1).to_numpy()] += home_team_wins['Winner'].values
            for league in leagues:
                league_df = self.df.loc[self.df['League'] == league]
                teams_ids = pd.concat([league_df['Home'], league_df['Away']], axis=0).unique()
                league_teams_score = team_points[teams_ids]
                teams_sorted = np.argsort(league_teams_score.astype(int))
                teams_sorted = np.max(teams_sorted) - teams_sorted
                team_ranks[teams_ids] += teams_sorted
                team_ranks[teams_ids] = normalize_array(team_ranks[teams_ids])
            labels.append(team_ranks)

            start_date += delta

        return edges, edge_features, labels

    def get_dataset(self, node_f_extract: bool, node_f_discount: float = 0.75, node_f_draws: bool = False,
                  verbose: bool = False) -> DynamicGraphTemporalSignal:
        """ transform dataframe into dataset used by PyTorch Geometric
            after transformation, mapping of the teams to unique ids is saved to self.mapping
            the transformation is destructive for the original dataframe

            :param node_f_extract: (bool) extract features from nodes

            :keyword node_f_discount: (float) discount factor for node features
            :keyword node_f_draws: (bool) use draws in features
            :keyword verbose: (bool) print log to console
        """

        # 1) map teams to unique ids
        self._create_teams_mapping()

        # 2) map match outcomes to integers (win home = 0; win away = 2; draw = 1)
        self._map_match_outcomes(verbose)

        # 3) extract node features (optional)
        if node_f_extract:
            node_features = self._extract_node_features(node_f_discount, node_f_draws, verbose)
        else:
            node_features = [None for _ in range(self.snapshot_count)]

        # 4) extract edges and edge features
        edges, edge_features, labels = self._extract_dynamic_edges_and_labels()

        # 5) create PyTorch Geometric Temporal dataset
        dynamic_graph = DynamicGraphTemporalSignal(
            node_features=node_features,
            edge_indices=edges,
            edge_weights=edge_features,
            features=node_features,
            targets=labels
        )

        return dynamic_graph
