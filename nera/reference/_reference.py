from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger


class RatingReference:
    def __init__(self, matches: pd.DataFrame, team_mapping: dict):
        """
        Class RatingReference
        Compute rating manually from matches dataframe
        :param matches: (DataFrame)
        :param team_mapping: (dict) mapping of teams to their ids
        """
        self._matches = (matches, team_mapping)

    @property
    def matches(self):
        """ Matches property only returns the DataFrame """
        return self._matches[0]

    @matches.setter
    def matches(self, value: tuple[pd.DataFrame, dict]):
        _error_string = "Matches setter must be a tuple of DataFrame and dict"
        if not isinstance(value, tuple):
            raise TypeError(_error_string)
        if len(value) != 2:
            raise ValueError(_error_string)

        matches, team_mapping = value
        if not isinstance(matches, pd.DataFrame):
            raise TypeError(_error_string)
        if not isinstance(team_mapping, dict):
            raise TypeError(_error_string)

        self._matches = (matches, team_mapping)

    def compute_reference(self, rating: str, **kwargs) -> Sequence[np.ndarray]:
        """
        :param rating: (str) 'elo' / 'berrar' / 'pi'
        :return: Sequence[ndarray] computed rating values
        """
        match rating:
            case 'elo':
                return self._elo_reference(**kwargs)
            case 'berrar':
                raise NotImplementedError
            case 'pi':
                raise NotImplementedError
            case _:
                logger.error(f'Unknown rating {rating}')
                raise ValueError(f'Unknown rating {rating}')

    def _elo_reference(self, elo_base: int = 1000, gamma: float = 2, c: float = 3, d: float = 500,
                       k: float = 3, verbose: bool = False, **kwargs) -> Sequence[np.ndarray]:
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
        matches, mapping = self._matches
        elo = np.zeros((len(mapping),)) + elo_base
        E_HS = []
        elos = []
        for i in range(len(matches.index)):
            match_i = matches.iloc[i]
            h_i = mapping[match_i['Home']]
            a_i = mapping[match_i['Away']]

            E_h = 1 / (1 + np.power(c, ((elo[a_i] - elo[h_i]) / d)))
            S_h = 1. if match_i['Winner'] == 'home' else 0. if match_i['Winner'] == 'away' else 1 / 2

            E_HS.append(E_h)

            elos_b4 = (elo[h_i], elo[a_i])

            h_points = match_i['Home_points']
            a_points = match_i['Away_points']

            delta = abs(h_points - a_points)
            update = k * ((1 + delta) ** gamma) * (S_h - E_h)

            elo[h_i] += update
            elo[a_i] -= update

            elos.append((elos_b4, (elo[h_i], elo[a_i])))

            if verbose:
                print(f"iteration {i}, rating: {elo}, E_H = {E_h}; {S_h}")
        return [np.array(elo)]
