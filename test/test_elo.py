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
        self.team_count = random.randint(4, 20)
        self.number_of_seasons = random.randint(1, 10)
        self.matches_per_season = random.randint(10, 100)

        self.numerical = EloNumerical(team_count=self.team_count)
        self.analytical = EloAnalytical(team_count=self.team_count)
        self.manual = EloManual(team_count=self.team_count)

        self.dataset = generate_random_matches(self.team_count, self.matches_per_season, self.number_of_seasons)

        self.transform = DataTransformation(self.dataset, timedelta(days=365))
        self.temporal_dataset = self.transform.get_dataset()

        self.reference_maker = RatingReference(self.dataset, self.transform.team_mapping)

    def test_init(self) -> None:
        self.assertEqual(self.numerical.is_rating, True)
        self.assertEqual(self.analytical.is_rating, True)
        self.assertEqual(self.manual.is_rating, True)

        self.assertEqual(self.numerical.is_manual, False)
        self.assertEqual(self.analytical.is_manual, False)
        self.assertEqual(self.manual.is_manual, True)

    def test_manual(self) -> None:
        trainer = Trainer(self.temporal_dataset, self.manual, train_ratio=1)
        trainer.train(epochs=1)

        manual_rating = self.manual.elo.detach()
        reference = self.reference_maker.compute_reference('elo')[0]

        np.testing.assert_array_equal(manual_rating, reference)

    def test_gradient(self):
        trainer = Trainer(self.temporal_dataset, train_ratio=1)
        trainer.model = self.analytical
        trainer.train(epochs=1)

        trainer.model = self.numerical
        trainer.train(epochs=1)

        numerical = self.numerical.elo.detach()
        analytical = self.analytical.elo.detach()

        np.testing.assert_array_almost_equal(numerical, analytical, decimal=3)


if __name__ == '__main__':
    unittest.main()
