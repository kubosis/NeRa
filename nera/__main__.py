#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NERA - Neural Ratings Analysis

simple file for executing nera directly via command line
"""

import argparse
import sys

from .data import DataAcquisition, FROM_CSV, FROM_FLASHSCORE, FROM_NBA_STATS
from .utils import process_kwargs


def acquire_data(args):
    # from where to acquire data
    da = DataAcquisition()
    kwargs = {}
    if args.from_csv:
        from_flag = FROM_CSV
        kwargs['fname'] = args.csv_fpath_from
        da.get_data(from_flag, **kwargs)
    elif args.from_nba_stats:
        date_from: str = "01/01/1990"
        date_to: str = "01/20/1993"
        kwargs['date_from'] = date_from
        kwargs['date_to'] = date_to
        from_flag = FROM_NBA_STATS
        da.get_data(from_flag, **kwargs)
    elif args.from_flashscore:
        country = 'europe'
        league = 'EuroCup'

        # actual data
        league_years = "2023-2024"
        kwargs['url'] = f'https://www.flashscore.com/basketball/{country}/{league.lower()}/results/'
        kwargs['year'] = league_years
        kwargs['state'] = 'EU'
        kwargs['league'] = league
        kwargs['keep_df'] = True
        from_flag = FROM_FLASHSCORE
        da.get_data(from_flag, **kwargs)

        # archived data
        for year in range(2023, 2012, -1):
            league_years = str(year - 1) + "-" + str(year)
            kwargs['url'] = f'https://www.flashscore.com/basketball/{country}/{league.lower()}-{league_years}/results/'
            kwargs['year'] = league_years
            kwargs['league'] = league
            from_flag = FROM_FLASHSCORE
            da.get_data(from_flag, **kwargs)
    else:
        raise RuntimeError("set from where you want to get your data; run with -h for more info")

    if args.csv_store:
        da.safe_data_csv("other_leagues.csv")
    if args.database_store:
        process_kwargs(["dbs_pwd"], **vars(args))
        da.save_data_to_database(ssh_host=args.ssh_host,
                                 ssh_user=args.ssh_user,
                                 ssh_pkey=args.ssh_pkey,
                                 db_name=args.dbs_name,
                                 table=args.dbs_table,
                                 schema=args.dbs_schema,
                                 db_user=args.dbs_user,
                                 db_pwd=args.dbs_pwd)


def parse_args():
    parser = argparse.ArgumentParser()
    # data acquiring flags --------------------------------------------------------------------------
    parser.add_argument('-a', '--acquire-data', action='store_true',
                        help="Main program flow flag - enables program flow for data handling")
    parser.add_argument('-fc', '--from-csv', action='store_true',
                        help="Get data from csv file, specify filepath in csv-fpath arg; no effect when -a not set")
    parser.add_argument('--csv-fpath-from', type=str,
                        help="Specify csv filepath; no effect when -a not set")
    parser.add_argument('-fn', '--from-nba-stats', action='store_true',
                        help="Get data from nba stats; no effect when -a not set")
    parser.add_argument('-ff', '--from-flashscore', action='store_true',
                        help="Get data from flashscore; no effect when -a not set")
    parser.add_argument('-cs', '--csv-store', action='store_true',
                        help="Store acquired data to csv file, specify fpath in csv-fpath-to; no effect when -a not set")
    parser.add_argument('--csv-fpath-to', type=str,
                        help="Specify csv filepath; no effect when -a not set")
    parser.add_argument('-db', '--database-store', action='store_true',
                        help="Store acquired data on school database; no effect when -a not set")
    parser.add_argument('-dbp', '--dbs-pwd', type=str,
                        help="Password for postgres database. "
                             "Has to be set when -db is set; no effect when -a not set")
    parser.add_argument('-dbu', '--dbs-user', type=str,
                        help="Database username "
                             "Has to be set when -db is set; no effect when -a not set")
    parser.add_argument('--dbs-name', type=str,
                        help="Name of the database "
                             "Has to be set when -db is set; no effect when -a not set")
    parser.add_argument('--dbs-schema', type=str,
                        help="Schema to witch you want to save data within database "
                             "Has to be set when -db is set; no effect when -a not set")
    parser.add_argument('--dbs-table', type=str,
                        help="Table to witch you want to save data within database "
                             "Has to be set when -db is set; no effect when -a not set")
    parser.add_argument('--ssh-host', type=str,
                        help="SSH host DNS or public IP. "
                             "Has to be set when -db is set; no effect when -a not set")
    parser.add_argument('--ssh-user', type=str,
                        help="SSH username"
                             "Has to be set when -db is set; no effect when -a not set")
    parser.add_argument('--ssh-pkey', type=str,
                        help="path to ssh private key"
                             "Has to be set when -db is set; no effect when -a not set")
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
