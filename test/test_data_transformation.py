import unittest
from datetime import timedelta
import random

import pandas as pd

from nera.data import DataTransformation
from nera.data.utils import generate_random_matches


class TestDataTransformation(unittest.TestCase):
    def setUp(self):
        self.team_count = 8
        self.number_of_seasons = 10
        self.matches_per_season = 100

        x = self.dataset = generate_random_matches(self.team_count, self.matches_per_season, self.number_of_seasons)
        self.dataset['DT'] = pd.to_datetime(self.dataset['DT'], format="%Y-%m-%d %H:%M:%S")
        self.dataset = self.dataset.sort_values(by='DT', ascending=False)
        self.transform = DataTransformation(self.dataset, timedelta(days=365))
        self.temporal_dataset = self.transform.get_dataset()

        self.mapping = self.transform.team_mapping
        self.inv_mapping = self.transform.inv_team_mapping

    def test_temporal_dataset(self):
        idx = len(self.dataset) - 1
        for (time, snapshot) in enumerate(self.temporal_dataset):
            matches = snapshot.edge_index
            match_points = snapshot.match_points
            outcomes = snapshot.edge_attr

            for m in range(matches.shape[1]):
                match = matches[:, m]
                outcome = outcomes[m, :]
                home_pts, away_pts = match_points[m, 0], match_points[m, 1]

                self.assertEqual(home_pts, self.dataset.iloc[idx]['Home_points'])
                self.assertEqual(away_pts, self.dataset.iloc[idx]['Away_points'])

                home_id, away_id = match
                home_id = int(home_id)
                away_id = int(away_id)
                home, away = self.inv_mapping[home_id], self.inv_mapping[away_id]

                self.assertEqual(home, self.dataset.iloc[idx]['Home'])
                self.assertEqual(away, self.dataset.iloc[idx]['Away'])

                outcome_str = 'home' if outcome == 1 else 'away' if outcome == 0 else 'draw'

                self.assertEqual(outcome_str, self.dataset.iloc[idx]['Winner'])

                idx -= 1


if __name__ == '__main__':
    unittest.main()
