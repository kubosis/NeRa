import json
import pandas as pd
import pickle
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder

from ._macros import PATH

__all__ = [
    "save_to_pickle",
    "load_from_pickle",
    "save_json",
    "load_json",
    "safe_data_csv",
    "load_data_csv",
    "ssh_save_data_to_postgres",
]

from .utils import ssh_tunnel


def save_to_pickle(fname: str, data):
    """
    Saving data to pickle
    """
    with open(fname, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"data saved to pickle {fname}")


def load_from_pickle(fname: str):
    """
    Loading data from pickle
    """
    with open(fname, "rb") as file:
        data = pickle.load(file)
        logger.info(f"{fname} loaded from pickle")
        return data


def save_json(
    df: pd.DataFrame | dict, fname: str = "data.json", fpath: str = PATH
) -> None:
    if isinstance(df, dict):
        json_obj = json.dumps(df, indent=4)
    else:
        json_obj = json.loads(df.to_json(orient="index"))

    with open(fpath + fname, "w") as outfile:
        outfile.write(json_obj)
    logger.info(f"df saved to json {fname}")


def load_json(fname: str = "data.json", fpath: str = PATH) -> pd.DataFrame:
    with open(fpath + fname, "r") as openfile:
        json_obj = json.load(openfile)
    df = pd.DataFrame.from_dict(json_obj, orient="index")
    logger.info(f"data loaded from json {fname}")
    return df


def safe_data_csv(fname: str, df: pd.DataFrame, fpath: str = PATH) -> None:
    filepath = Path(fpath + fname)
    init = not filepath.is_file()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if init:
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, index=False, mode="a", header=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")


def load_data_csv(fname: str):
    df = pd.read_csv(fname)
    logger.info(f"{len(df)} rows loaded from {fname}")
    return df


@ssh_tunnel
def ssh_save_data_to_postgres(
    df: pd.DataFrame,
    db_name: str,
    table: str,
    schema: str,
    db_user: str,
    db_pwd: str,
    ssh_server: SSHTunnelForwarder,
) -> None:
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
    local_port = 5432  # str(ssh_server.local_bind_port)
    host = "localhost"
    connect_string = f"postgresql://{db_user}:{db_pwd}@{host}:{local_port}/{db_name}"
    engine = create_engine(
        connect_string,
        pool_pre_ping=True,
    )
    logger.info(f"Postgres engine created for {connect_string}")

    df.to_sql(table, engine, schema=schema)
    logger.info("dataframe saved to databse")
