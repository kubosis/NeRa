import argparse
import sys

from src.Data_mgmt import DataAcquisition, FROM_CSV, FROM_NBA_STATS


def acquire_data(from_flag: int, db_pwd: str = "", to_csv: bool = False, to_db: bool = False):
    da = DataAcquisition()
    date_from: str = "01/01/1990"
    date_to: str = "01/20/1993"
    da.get_nba_game_data(from_flag, "./resources/nba.csv", date_from, date_to)
    if to_csv:
        da.safe_data_csv("nba.csv")
    if to_db:
        da.save_data_to_database(ssh_host="potato.felk.cvut.cz",
                                 ssh_user="sukdojak",
                                 ssh_pkey="~/.ssh/id_ed25519",
                                 db_name="students",
                                 table="NBA",
                                 schema="basketball",
                                 db_user="sukdojak",
                                 db_pwd=db_pwd)


def main():
    # -- parse args for program flow control
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--acquire-data', action='store_true',
                        help="Main program flow flag - enables program flow for data handling")
    parser.add_argument('-db', '--database-store', action='store_true',
                        help="Store acquired data on school database; no effect when -ad not set")
    parser.add_argument('-cs', '--csv-store', action='store_true',
                        help="Store acquired data to csv file; no effect when -ad not set")
    parser.add_argument('-fc', '--from-csv', action='store_true',
                        help="Get data from csv file; no effect when -ad not set")
    parser.add_argument('-fn', '--from-nba-stats', action='store_true',
                        help="Get data from nba stats; no effect when -ad not set")
    parser.add_argument('-dbp', '--dbs-pwd', type=str,
                        help="Password for postgres database. "
                             "Has to be set when -db is set; no effect when -ad not set")
    args = parser.parse_args()

    # -- run desired code
    if args.acquire_data:
        if args.from_csv:
            from_flag = FROM_CSV
        elif args.from_nba_stats:
            from_flag = FROM_NBA_STATS
        else:
            raise RuntimeError("set from where you want to get your data")
        dbs_pwd = args.dbs_pwd if "dbs_pwd" in args else ""
        acquire_data(from_flag, dbs_pwd, args.csv_store, args.database_store)
    else:
        ...


if __name__ == "__main__":
    main()
    sys.exit(0)
