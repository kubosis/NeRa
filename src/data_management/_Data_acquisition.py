"""
    Name: _Data_acquisition.py
    Author: Jakub Sukdol
    Date: 25.10.23

    Basic data acquisition class and functions
"""

__all__ = ["DataAcquisition", "save_data_to_database"]

import pandas as pd
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
from loguru import logger
from nba_api.stats.endpoints import leaguegamefinder
from typing import Optional

from src.utils.wrappers import ssh_tunnel
from src.data_management._macros import _PATH, FROM_CSV, FROM_NBA_STATS
from src.data_management._data_saving_loading import *


@ssh_tunnel
def save_data_to_database(df: pd.DataFrame, db_name: str, table: str, schema: str,
                          db_user: str, db_pwd: str, ssh_server: SSHTunnelForwarder) -> None:
    """
    Connect  to postgres database via SSH tunnelling and create table from df
    :param df: (pandas.Dataframe)
    :param db_name: (str) name of postgres database
    :param table: (str) name of table to be created / appended
    :param schema: (str) name of postgres schema
    :param db_user: (str) postgres db username
    :param db_pwd: (str) postgres db password
    :param ssh_server: ADDED BY DECORATOR! Don't include in function call as it will have no effect
    :return: (None)

    ! IMPORTANT !
    for Keyword arguments for ssh tunnel see documentation of the decorator
    """
    if df is None:
        raise ValueError("Cannot push NoneType dataframe")

    # connect to PostgreSQL
    local_port = str(ssh_server.local_bind_port)
    connect_string = f'postgresql://{db_user}:{db_pwd}@localhost:{local_port}/{db_name}'
    engine = create_engine(connect_string)
    logger.info(f"Postgres engine created for {connect_string}")

    df.to_sql(table, engine, schema=schema)
    logger.info("dataframe saved to databse")


class DataAcquisition:
    def __init__(self, out_path: str = _PATH):
        self.fpath: str = out_path
        self.df: Optional[pd.DataFrame] = None

    def _data_nba_by_date(self, date_from: str, date_to: str) -> None:
        game_finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_from, date_to_nullable=date_to, timeout=60)
        games = game_finder.get_data_frames()[0]
        self.df = games

    def _get_data_from_csv(self, fname, fpath):
        self.df = load_data_csv(fname, fpath)

    def get_nba_game_data(self, o_from: int, fname: str, fpath: str = _PATH,
                          date_from: str = "01/01/1990", date_to: str = "01/01/2023") -> pd.DataFrame:
        if o_from & FROM_NBA_STATS:
            self._data_nba_by_date(date_from, date_to)
        elif o_from & FROM_CSV:
            self._get_data_from_csv(fname, fpath)
        else:
            raise ValueError("Wrong Option flag o_from set")

        return self.df

    def save_data_to_database(self, *args, **kwargs):
        """ reference for simpler manipulation (for details see save_data_to_database doc) """
        save_data_to_database(*args, df=self.df, **kwargs)

    def safe_data_csv(self, fname: str):
        """ reference for simpler manipulation """
        safe_data_csv(df=self.df, fname=fname)
