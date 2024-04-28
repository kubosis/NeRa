from typing import Any
import random
from datetime import timedelta, datetime
import copy
from contextlib import contextmanager

import torch.nn as nn
import torch
import pandas as pd
import numpy as np

__all__ = [
    "process_kwargs",
    "print_rating_diff",
    "generate_random_matches",
    "conditional_nograd_context",
    "one_hot_encode",
    "normalize_array",
]


def process_kwargs(mandatory: list, **kwargs) -> tuple[list[Any], dict[str, Any]]:
    """check kwargs and return list of popped values and updated kwargs"""
    ret_list = []
    for elem in mandatory:
        if elem not in kwargs:
            raise ValueError(f"Mandatory keyword argument {elem} omitted")
        ret_list.append(kwargs.pop(elem))
    return *ret_list, kwargs


def print_rating_diff(
    rating1: nn.Module,
    rating2: nn.Module,
    transform,
    rat_1_name: str = "Numerical",
    rat_2_name: str = "Symbolical",
    eps: float = 1e-2,
):
    assert (
        hasattr(rating1, "is_rating")
        and rating1.is_rating
        and hasattr(rating2, "is_rating")
        and rating2.is_rating
    )

    numerical = rating1
    analytical = rating2

    err = False
    err_sum = [0 * len(rating1.ratings)]
    err_count = [0 * len(rating1.ratings)]
    diff_percentage = [0 * len(rating1.ratings)]
    max_diff = [0 * len(rating1.ratings)]
    print("Ratings diff:")
    for rat in range(len(numerical.ratings)):
        print(f"Rating {rat}:")
        print_count = 0
        rating_num_i = numerical.ratings[rat]
        rating_ana_i = analytical.ratings[rat]
        for i in range(len(rating_num_i)):
            auto = float(
                rating_num_i[transform.team_mapping[transform.inv_team_mapping[i]]]
            )
            grad = float(
                rating_ana_i[transform.team_mapping[transform.inv_team_mapping[i]]]
            )
            diff = abs(auto - grad)
            if diff > eps:
                err = True
                diff_percentage[rat] += diff / ((abs(auto) + abs(grad)) / 2)
                err_sum[rat] += diff
                err_count[rat] += 1
                max_diff[rat] = max(diff, max_diff[rat])
            if print_count < 5:
                str_i = (
                    rf"[ERROR] in rating {rat} on index {i}:: {auto} / {grad}"
                    if err
                    else f"{i}:: computed: {auto:10.3f} || net: {grad:10.3f}"
                )
                print(str_i)
                print_count += 1

        print("...")
        print()

        if not err:
            print(
                "[SUCCESS]: All  ratings computed analytically and numerically are the SAME"
            )
        else:
            print(
                f"Total number of errors: {err_count[rat]} out of {len(rating_num_i)} computed ratings"
            )
            print(f"Cumulative sum of errors: {err_sum[rat]}")
            print(f"Average difference: {err_sum[rat] / err_count[rat]}")
            print(
                f"Average difference percentage: {diff_percentage[rat] / err_count[rat] * 100}%"
            )
            print(f"Maximal difference: {max_diff[rat]}")
        print("-------------------------")

    print()
    print("Hyperparams diff:")
    for hp in range(len(rating1.hyperparams)):
        num_hp_i = numerical.hyperparams[hp]
        ana_hp_i = analytical.hyperparams[hp]
        print(
            f"{rat_1_name}: hyperparam[{hp}] = {float(num_hp_i):8.3f} :: "
            f"{rat_1_name}: hyperparam[{hp}] = {float(ana_hp_i):8.3f}"
        )


def generate_random_matches(
    team_count: int, matches_per_season: int, season_count: int
) -> pd.DataFrame:
    team_names = [f"team_{i}" for i in range(team_count)]
    match_count = matches_per_season * season_count

    def generate_sequence(length, elements):
        if len(elements) < 2 and length > 1:
            raise ValueError(
                "Need at least two distinct elements for sequences longer than 1"
            )

        sequence = [random.choice(elements)]

        for _ in range(1, length):
            # Select a random element that is not the same as the last element in the sequence
            next_element = random.choice([el for el in elements if el != sequence[-1]])
            sequence.append(next_element)

        return sequence

    home_teams = generate_sequence(match_count, team_names)
    # made so that never the same team plays with itself
    away_teams = copy.deepcopy(home_teams[1:])
    away_teams.append(home_teams[0])

    winners = []
    home_points = []
    away_points = []
    for _ in range(len(home_teams)):
        rand_home = random.randint(0, 100)
        rand_away = random.randint(0, 100)
        winner = (
            "home"
            if rand_home > rand_away
            else "away"
            if rand_home < rand_away
            else "draw"
        )
        winners.append(winner)
        home_points.append(rand_home)
        away_points.append(rand_away)

    dt = []
    delta1 = timedelta(seconds=1)
    delta2 = timedelta(days=366)
    now = datetime.now()
    for i in range(season_count):
        dt.extend([*[now - i * delta2 + j * delta1 for j in range(matches_per_season)]])

    data = pd.DataFrame(
        {
            "DT": dt,
            "Home": home_teams,
            "Away": away_teams,
            "Winner": winners,
            "Home_points": home_points,
            "Away_points": away_points,
            "League": [*(match_count * ["liga"])],
        }
    )
    data = data.sort_values(by="DT", ascending=False)
    return data


@contextmanager
def conditional_nograd_context(nograd: bool):
    """
    Context manager to conditionally apply a context.

    Parameters:
    - condition (bool): If True, applies torch.no_grad() context.
    """
    if nograd:
        with torch.no_grad():
            yield
    else:
        yield


def normalize_array(array: np.ndarray) -> np.ndarray:
    denominator = np.max(array) - np.min(array)
    denominator = 1 if denominator == 0 else denominator
    normalized = (array - np.min(array)) / denominator
    return normalized


def one_hot_encode(i: int, max_n: int) -> np.ndarray:
    out = np.zeros((max_n,))
    out[i] = 1
    return out
