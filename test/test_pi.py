import unittest
from datetime import timedelta
import random

import numpy as np
import torch

from nera.models.ratings import PiManual
from nera.reference import RatingReference
from nera.trainer import Trainer
from nera.data.utils import generate_random_matches
from nera.data import DataTransformation


class TestBerrar(unittest.TestCase):
    def setUp(self) -> None:
        self._params = {
            'lambda_': torch.tensor(0.001, dtype=torch.float64),
            'gamma': torch.tensor(0.01, dtype=torch.float64),
            'c': torch.tensor(20, dtype=torch.float64),
            'default': 1000
        }

        self.team_count = random.randint(10, 20)
        self.number_of_seasons = random.randint(10, 20)
        self.matches_per_season = random.randint(100, 1000)

        self.manual = PiManual(team_count=self.team_count, **self._params)

        self.dataset = generate_random_matches(self.team_count, self.matches_per_season, self.number_of_seasons)

        self.transform = DataTransformation(self.dataset, timedelta(days=365))
        self.temporal_dataset = self.transform.get_dataset()

        self.reference_maker = RatingReference(self.transform.num_teams)

    def test_init(self):
        print('Testing init...')
        self.assertEqual(self.manual.is_rating, True)

        self.assertEqual(self.manual.is_manual, True)

    def test_manual(self):
        print('Testing manual...')
        trainer = Trainer(self.temporal_dataset, self.manual, train_ratio=1)
        trainer.train(epochs=1, val_ratio=0)

        manual_rating = self.manual.ratings
        reference = self.reference_maker.compute_reference('pi', self.temporal_dataset, **self._params)

        for r in range(len(manual_rating)):
            man = manual_rating[r].detach()
            np.testing.assert_array_almost_equal(man, reference[r], decimal=3)
            print(f'Manual Rating:\n{manual_rating[r]};\nReference Rating:\n{reference[r]}')


if __name__ == '__main__':
    unittest.main()
