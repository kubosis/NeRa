"""
    Name: _Data_acquisition.py
    Author: Jakub Sukdol
    Date: 25.10.23

    Basic data acquisition classes and functions
    No data preparation and cleaning
"""

__all__ = ["FROM_CSV", "FROM_NBA_STATS", "DataAcquisition"]

import pandas as pd
import json
from nba_api.stats.static import teams
from pathlib import Path
from nba_api.stats.endpoints import leaguegamefinder
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
from loguru import logger

# Macros and definitions
_PATH: str = "./resources/"

FROM_CSV = 1 << 0
FROM_NBA_STATS = 1 << 1


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

    def _data_nba_by_date(self, date_from: str, date_to: str) -> None:
        game_finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_from, date_to_nullable=date_to, timeout=60)
        games = game_finder.get_data_frames()[0]
        self.df = games

    def get_nba_game_data(self, o_from: int, fpath: str = "",
                          date_from: str = "01/01/1990", date_to: str = "01/01/2023") -> None:
        if o_from & FROM_NBA_STATS:
            self._data_nba_by_date(date_from, date_to)
        elif o_from & FROM_CSV and fpath != "":
            self.df = pd.read_csv(fpath)

    def save_data_to_database(self, ssh_host: str, ssh_user: str, ssh_pkey: str,
                              db_name: str, table: str, schema: str, db_user: str, db_pwd: str) -> None:
        """
        Connect  to postgres database via SSH tunnelling and create table from df
        :param ssh_host: (str) host ssh server
        :param ssh_user: (str) ssh username
        :param ssh_pkey: (str) path to ssh private key
        :param db_name: (str) name of postgres database
        :param table: (str) name of table to be created / appended
        :param schema: (str) name of postgres schema
        :param db_user: (str) postgres db username
        :param db_pwd: (str) postgres db password
        :return: (None)
        """
        if self.df is None:
            raise ValueError("No dataframe created")

        with SSHTunnelForwarder(
                (ssh_host, 22),
                ssh_username=ssh_user,
                ssh_pkey=ssh_pkey,
                remote_bind_address=('127.0.0.1', 5432)
        ) as server:
            server.start()  # start ssh sever
            logger.info("Server connected via ssh")

            # connect to PostgreSQL
            local_port = str(server.local_bind_port)
            connect_string = f'postgresql://{db_user}:{db_pwd}@localhost:{local_port}/{db_name}'
            engine = create_engine(connect_string)
            logger.info(f"Postgres engine created for {connect_string}")

            self.df.to_sql(table, engine, schema=schema)
            logger.info("dataframe saved to databse")

    def safe_data_csv(self, fname: str) -> None:
        filepath = Path(self.fpath + fname)
        init = not filepath.is_file()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if init:
            self.df.to_csv(filepath, index=False)
        else:
            self.df.to_csv(filepath, index=False, mode='a', header=False)
