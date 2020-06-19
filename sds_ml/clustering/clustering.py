import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import argparse
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
SILENT = 0

log = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser(description="SDS clustering examples")

    parser.add_argument("name", type=str, help="Name of the example to run")

    args = parser.parse_args()

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()

