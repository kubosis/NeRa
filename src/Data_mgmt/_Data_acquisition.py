"""
    Name: _Data_acquisition.py
    Author: Jakub Sukdol
    Date: 25.10.23

    Basic data acquisition classes and functions
    No data preparation and cleaning
"""

import pandas as pd
import json
from nba_api.stats.static import teams
from pathlib import Path
from nba_api.stats.endpoints import leaguegamefinder


# Macros and definitions
_PATH: str = "./resources/"


def save_json(df: pd.DataFrame | dict, fname: str = "data.json", fpath: str = _PATH) -> None:
    if isinstance(df, dict):
        json_obj = json.dumps(df, indent=4)
    else:
        json_obj = json.loads(df.to_json(orient="index"))

    with open(fpath + fname, "w") as outfile:
        outfile.write(json_obj)


def load_json(fname: str = "data.json", fpath: str = _PATH) -> pd.DataFrame:
    with open(fpath + fname, 'r') as openfile:
        json_obj = json.load(openfile)
    df = pd.DataFrame.from_dict(json_obj, orient="index")
    return df


class DataAcquisition:
    def __init__(self, out_path: str = _PATH):
        self.fpath: str = out_path
        self.df: pd.DataFrame | None = None

    def get_nba_data(self) -> None:
        gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable="01/01/1990", date_to_nullable="01/01/2023")
        games = gamefinder.get_data_frames()[0]
        self.df = games

    def safe_data_csv(self, fname: str):
        filepath = Path(self.fpath + fname)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(filepath)


