import unittest
from datetime import timedelta
import random

import numpy as np

from nera.models.ratings import EloNumerical, EloManual, EloAnalytical
from nera.reference import RatingReference
from nera.trainer import Trainer
from nera.utils import generate_random_matches
from nera.data import DataTransformation


class TestElo(unittest.TestCase):
    def setUp(self) -> None:
        self._elo_params = {
            'k': 3,
            'gamma': 0.6,
            'c': 2,
            'd': 500
        }

        self.team_count = random.randint(10, 20)
        self.number_of_seasons = random.randint(10, 20)
        self.matches_per_season = random.randint(100, 1000)

        self.numerical = EloNumerical(team_count=self.team_count, hp_grad=True, **self._elo_params)
        self.analytical = EloAnalytical(team_count=self.team_count, hp_grad=True, **self._elo_params)
        self.manual = EloManual(team_count=self.team_count, **self._elo_params)

        self.dataset = generate_random_matches(self.team_count, self.matches_per_season, self.number_of_seasons)

        self.transform = DataTransformation(self.dataset, timedelta(days=365))
        self.temporal_dataset = self.transform.get_dataset()

        self.reference_maker = RatingReference(self.transform.num_teams)

    def test_init(self) -> None:
        print('Testing init...')
        self.assertEqual(self.numerical.is_rating, True)
        self.assertEqual(self.analytical.is_rating, True)
        self.assertEqual(self.manual.is_rating, True)

        self.assertEqual(self.numerical.is_manual, False)
        self.assertEqual(self.analytical.is_manual, False)
        self.assertEqual(self.manual.is_manual, True)

    def test_manual(self) -> None:
        print('Testing manual...')
        trainer = Trainer(self.temporal_dataset, self.manual, train_ratio=1)
        trainer.train(epochs=1, val_ratio=0)

        manual_rating = self.manual.elo.detach()
        reference = self.reference_maker.compute_reference('elo', self.temporal_dataset, **self._elo_params)[0]

        np.testing.assert_array_almost_equal(manual_rating, reference, decimal=3)
        print(f'Manual Rating:\n{manual_rating};\nReference Rating:\n{reference}')

    def test_gradient(self):
        print('Testing gradient...')
        trainer = Trainer(self.temporal_dataset, train_ratio=1)
        trainer.model = self.analytical
        trainer.train(epochs=1)

        trainer.model = self.numerical
        trainer.train(epochs=1)

        numerical = self.numerical.elo.detach()
        analytical = self.analytical.elo.detach()

        np.testing.assert_array_almost_equal(numerical, analytical, decimal=1)
        print(f'Numerical rating:\n{numerical};\nAnalytical rating:\n{analytical}')

        numerical_hp = self.numerical.hyperparams
        analytical_hp = self.analytical.hyperparams
        for i in range(len(numerical_hp)):
            self.assertAlmostEqual(float(numerical_hp[i]), float(analytical_hp[i]), places=3)
        print(f'Numerical hyperparams:\n{numerical_hp};\nAnalytical hyperparams:\n{analytical_hp}')


if __name__ == '__main__':
    unittest.main()
