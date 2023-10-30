__all__ = ["check_input"]


def check_input(mandatory: list, **kwargs):
    for elem in mandatory:
        if elem not in kwargs:
            raise ValueError(f"Mandatory keyword argument {elem} omitted")
