import argparse
import sys

from src.Data_mgmt import DataAcquisition


def acquire_data():
    da = DataAcquisition()
    da.get_nba_data()
    da.safe_data_csv("nba.csv")


def main():
    # -- parse args for program flow control
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--acquire-data', action='store_true')
    args = parser.parse_args()

    # -- run desired code
    if args.acquire_data:
        acquire_data()
    else:
        ...


if __name__ == "__main__":
    main()
    sys.exit(0)
