import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import csv
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
SILENT = 0

log = logging.getLogger(__name__)

def read_csv(path):

    with pathlib.Path(path).open() as f:

        reader = csv.reader(f, delimiter=",")

        yield from reader

def read_csv_to_tuples(schema, NamedTuple, path):

    yield from (
        NamedTuple(*(transform(value) for transform, value in zip(schema,row)))
        for row
        in read_csv(path)
    )

def load():

    return tuple(open())

def open():

    dataset_path = "sds_ml/datasets/pima-indians-diabetes.csv"

    def boolean(s):

        return bool(int(s))

    schema = (
        int,
        int,
        int,
        int,
        int,
        float,
        float,
        int,
        boolean,
    )

    Pima = collections.namedtuple("Pima", [
        "pregnancy_count", # Number of times pregnant.
        "glucose_concentration", # Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
        "blood_pressure", # Diastolic blood pressure (mm Hg).
        "triceps_skinfold_thickness", # Triceps skinfold thickness (mm).
        "insulin", # 2-Hour serum insulin (mu U/ml).
        "bmi", #Body mass index (weight in kg/(height in m)^2).
        "dpf", # Diabetes pedigree function.
        "age", # Age (years).
        "signs_of_diabetes", # Class variable (0 or 1).
    ])

    yield from read_csv_to_tuples(schema=schema, NamedTuple=Pima, path=dataset_path)

def main():

    pass



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()
