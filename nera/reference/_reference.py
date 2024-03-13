from typing import Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger


class RatingReference:
    def __init__(self, num_teams):
        """
        Class RatingReference
        Compute rating manually from matches dataframe
        :param num_teams: (int) number of teams
        """
        self.num_teams = num_teams

    def compute_reference(self, rating: str, temp_dataset, **kwargs) -> Sequence[np.ndarray]:
        """
        :param rating: (str) 'elo' / 'berrar' / 'pi'
        :param temp_dataset: (Temporal Dataset Signal Iterator)
        :return: Sequence[ndarray] computed rating values
        """
        if rating == 'elo':
            return self._elo_reference(temp_dataset, **kwargs)
        elif rating == 'berrar':
            return self._berrar_reference(temp_dataset, **kwargs)
        elif rating == 'pi':
            raise NotImplementedError
        else:
            logger.error(f'Unknown rating {rating}')
            raise ValueError(f'Unknown rating {rating}')

    def _elo_reference(self, temp_dataset, elo_base: int = 1000, gamma: float = 2., c: float = 3., d: float = 500.,
                       k: float = 3., verbose: bool = False, **kwargs) -> Sequence[np.ndarray]:
        """
        Compute elo rating manually from natches dataframe
        :param elo_base: (float) base elo value
        :param gamma: (float) hyperparameter
        :param c: (float) hyperparameter
        :param d: (float) hyperparameter
        :param k: (float) learning rate
        :param verbose: (bool) print
        :return: np.ndarray with elo values
        """

        elo = np.zeros((self.num_teams,)) + elo_base
        E_HS = []
        elos = []
        iter_ = 0
        for time, snapshot in enumerate(temp_dataset):
            matches = snapshot.edge_index
            match_points = snapshot.match_points

            for m in range(matches.shape[1]):
                match = matches[:, m]

                h_i, a_i = match
                home_pts, away_pts = match_points[m, 0], match_points[m, 1]

                E_h = 1 / (1 + np.power(c, ((elo[a_i] - elo[h_i]) / d)))
                S_h = 1. if home_pts > away_pts else 0. if home_pts < away_pts else 1 / 2

                E_HS.append(E_h)

                elos_b4 = (elo[h_i], elo[a_i])

                delta = abs(home_pts - away_pts)
                update = k * ((1 + delta) ** gamma) * (S_h - E_h)

                elo[h_i] += update
                elo[a_i] -= update

                elos.append((elos_b4, (elo[h_i], elo[a_i])))

                if verbose:
                    print(f"iteration {iter_}, rating: {elo}, E_H = {E_h}; {S_h}")
                iter_ += 1
        return [np.array(elo)]

    def _berrar_reference(self, temp_dataset, default: float = 1000, alpha_a: float = 180, alpha_h: float = 180,
                          beta_a: float = 2, beta_h: float = 2, bias_h: float = 0, bias_a: float = 0,
                          lr_a_att: float = 0.1, lr_h_att: float = 0.1,
                          lr_a_def: float = 0.1, lr_h_def: float = 0.1) -> Sequence[np.ndarray]:

        att_ = torch.zeros((self.num_teams,), dtype=torch.float64) + default
        def_ = torch.zeros((self.num_teams,), dtype=torch.float64) + default

        for time, snapshot in enumerate(temp_dataset):
            matches = snapshot.edge_index
            match_points = snapshot.match_points

            for m in range(matches.shape[1]):
                match = matches[:, m]

                h_i, a_i = match
                home_pts, away_pts = match_points[m, 0], match_points[m, 1]

                hatt, hdef = att_[h_i], def_[h_i]
                aatt, adef = att_[a_i], def_[a_i]

                # prediction
                ghat_h = alpha_h / (1 + torch.exp(-beta_h * (hatt + adef) - bias_h))
                ghat_a = alpha_a / (1 + torch.exp(-beta_a * (aatt + hdef) - bias_a))

                # update
                att_[h_i] += lr_h_att * (home_pts - ghat_h)
                def_[h_i] += lr_h_def * (away_pts - ghat_a)

                att_[a_i] += lr_a_att * (away_pts - ghat_a)
                def_[a_i] += lr_a_def * (home_pts - ghat_h)

        return [att_, def_]
