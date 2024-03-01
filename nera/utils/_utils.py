from typing import Any

import torch.nn as nn

__all__ = ["process_kwargs", "print_rating_diff"]


def process_kwargs(mandatory: list, **kwargs) -> tuple[list[Any], dict[str, Any]]:
    """ check kwargs and return list of popped values and updated kwargs """
    ret_list = []
    for elem in mandatory:
        if elem not in kwargs:
            raise ValueError(f"Mandatory keyword argument {elem} omitted")
        ret_list.append(kwargs.pop(elem))
    return *ret_list, kwargs


def print_rating_diff(rating1: nn.Module, rating2: nn.Module, transform,
                      rat_1_name: str = "Numerical", rat_2_name: str = "Symbolical", eps: float = 1e-2):

    assert hasattr(rating1, 'is_rating') and rating1.is_rating and hasattr(rating2, 'is_rating') and rating2.is_rating

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
            auto = float(rating_num_i[transform.team_mapping[transform.inv_team_mapping[i]]])
            grad = float(rating_ana_i[transform.team_mapping[transform.inv_team_mapping[i]]])
            diff = abs(auto - grad)
            if diff > eps:
                err = True
                diff_percentage[rat] += diff / (abs(auto + grad) / 2)
                err_sum[rat] += diff
                err_count[rat] += 1
                max_diff[rat] = max(diff, max_diff[rat])
            if print_count < 5:
                str_i = rf'[ERROR] in rating {rat} on index {i}:: {auto} / {grad}' if err \
                    else f'{i}:: computed: {auto:10.3f} || net: {grad:10.3f}'
                print(str_i)
                print_count += 1

        print("...")
        print()

        if not err:
            print("[SUCCESS]: All  ratings computed analytically and numerically are the SAME")
        else:
            print(f"Total number of errors: {err_count[rat]} out of {len(rating_num_i)} computed ratings")
            print(f"Cumulative sum of errors: {err_sum[rat]}")
            print(f"Average difference: {err_sum[rat] / err_count[rat]}")
            print(f"Average difference percentage: {diff_percentage[rat] / err_count[rat] * 100}%")
            print(f"Maximal difference: {max_diff[rat]}")
        print("-------------------------")

    print()
    print("Hyperparams diff:")
    for hp in range(len(rating1.hyperparams)):
        num_hp_i = numerical.hyperparams[hp]
        ana_hp_i = analytical.hyperparams[hp]
        print(f"{rat_1_name}: hyperparam[{hp}] = {float(num_hp_i):8.3f} :: "
              f"{rat_1_name}: hyperparam[{hp}] = {float(ana_hp_i):8.3f}")
