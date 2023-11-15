import argparse
import sys

from src.data_management import *
from src.utils import check_input


def acquire_data(args):
    # from where to acquire data
    da = DataAcquisition()
    kwargs = {}
    if args.from_csv:
        from_flag = FROM_CSV
        kwargs['fname'] = "basketball_other.csv"
    elif args.from_nba_stats:
        date_from: str = "01/01/1990"
        date_to: str = "01/20/1993"
        kwargs['date_from'] = date_from
        kwargs['date_to'] = date_to
        from_flag = FROM_NBA_STATS
    elif args.from_flashscore:
        kwargs['url'] = 'https://www.flashscore.com/basketball/czech-republic/nbl-2023-2024/results/'
        kwargs['year'] = '2023-2024'
        kwargs['state'] = 'CZ'
        kwargs['league'] = 'NBA'
        from_flag = FROM_FLASHSCORE
    else:
        raise RuntimeError("set from where you want to get your data; run with -h for more info")

    da.get_data(from_flag, **kwargs)
    if args.csv_store:
        da.safe_data_csv("other_leagues.csv")
    if args.database_store:
        check_input(["dbs_pwd"], **vars(args))
        da.save_data_to_database(ssh_host="potato.felk.cvut.cz",
                                 ssh_user="sukdojak",
                                 ssh_pkey="~/.ssh/id_ed25519",
                                 db_name="students",
                                 table="NBA",
                                 schema="basketball",
                                 db_user="sukdojak",
                                 db_pwd=args.dbs_pwd)


def parse_args():
    parser = argparse.ArgumentParser()
    # data acquiring flags --------------------------------------------------------------------------
    parser.add_argument('-a', '--acquire-data', action='store_true',
                        help="Main program flow flag - enables program flow for data handling")
    parser.add_argument('-db', '--database-store', action='store_true',
                        help="Store acquired data on school database; no effect when -a not set")
    parser.add_argument('-cs', '--csv-store', action='store_true',
                        help="Store acquired data to csv file; no effect when -a not set")
    parser.add_argument('-fc', '--from-csv', action='store_true',
                        help="Get data from csv file; no effect when -a not set")
    parser.add_argument('-fn', '--from-nba-stats', action='store_true',
                        help="Get data from nba stats; no effect when -a not set")
    parser.add_argument('-dbp', '--dbs-pwd', type=str,
                        help="Password for postgres database. "
                             "Has to be set when -db is set; no effect when -a not set")
    parser.add_argument('-ff', '--from-flashscore', action='store_true',
                        help="Get data from flashscore; no effect when -a not set")
    args = parser.parse_args()
    return args


def main():
    # -- parse args for program flow control
    args = parse_args()

    # -- run desired code
    if args.acquire_data:
        acquire_data(args)
    else:
        ...


if __name__ == "__main__":
    main()
    sys.exit(0)
