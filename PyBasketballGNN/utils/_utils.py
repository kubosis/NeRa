from typing import Any

__all__ = ["check_input"]


def check_input(mandatory: list, **kwargs) -> list[Any]:
    """ check kwargs and return list of popped values """
    ret_list = []
    for elem in mandatory:
        if elem not in kwargs:
            raise ValueError(f"Mandatory keyword argument {elem} omitted")
        ret_list.append(kwargs.pop(elem))
    return ret_list
