import json
import pandas as pd
import pickle
from pathlib import Path

from loguru import logger

from src.data_management._macros import PATH

__all__ = ["save_to_pickle", "load_from_pickle", "save_json", "load_json",
           "safe_data_csv", "load_data_csv"]


def save_to_pickle(fname: str, data):
    """
    Saving data to pickle
    """
    with open(fname, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(fname: str):
    """
    Loading data from pickle
    """
    with open(fname, 'rb') as file:
        data = pickle.load(file)
        return data


def save_json(df: pd.DataFrame | dict, fname: str = "data.json", fpath: str = PATH) -> None:
    if isinstance(df, dict):
        json_obj = json.dumps(df, indent=4)
    else:
        json_obj = json.loads(df.to_json(orient="index"))

    with open(fpath + fname, "w") as outfile:
        outfile.write(json_obj)


def load_json(fname: str = "data.json", fpath: str = PATH) -> pd.DataFrame:
    with open(fpath + fname, 'r') as openfile:
        json_obj = json.load(openfile)
    df = pd.DataFrame.from_dict(json_obj, orient="index")
    return df


def safe_data_csv(fname: str, df: pd.DataFrame, fpath: str = PATH) -> None:
    filepath = Path(fpath + fname)
    init = not filepath.is_file()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if init:
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, index=False, mode='a', header=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")


def load_data_csv(fname: str):
    df = pd.read_csv(fname)
    return df
