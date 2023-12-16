from typing import Any

__all__ = ["process_kwargs"]


def process_kwargs(mandatory: list, **kwargs) -> tuple[list[Any], dict[str, Any]]:
    """ check kwargs and return list of popped values and updated kwargs """
    ret_list = []
    for elem in mandatory:
        if elem not in kwargs:
            raise ValueError(f"Mandatory keyword argument {elem} omitted")
        ret_list.append(kwargs.pop(elem))
    return *ret_list, kwargs
