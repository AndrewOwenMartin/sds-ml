import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
import sds_ml.pima
SILENT = 0

log = logging.getLogger(__name__)

Result = collections.namedtuple("Result", ("truth","correct"))

def example_naive():

    rng = random.Random()

    dataset = sds_ml.pima.load()

    fields = next(iter(dataset))._fields

    evaluation = collections.Counter({
        Result(False, False):0,
        Result(False, True):0,
        Result(True, False):0,
        Result(True, True):0,
    })

    for row in dataset:

        truth = bool(row[-1])

        guess_is_correct = truth == False

        evaluation[(truth, guess_is_correct)] += 1

    positive = sum(1 for row in dataset if bool(row[-1]))

    positivity = positive/len(dataset)

    print(evaluation, f"Accuracy: {(evaluation[(False, True)] + evaluation[(True, True)])/len(dataset)}")

    print("positivity", positivity)

    evaluation = collections.Counter({
        Result(False, False):0,
        Result(False, True):0,
        Result(True, False):0,
        Result(True, True):0,
    })

    rng = random.Random()

    for row in dataset:

        truth = bool(row[-1])

        guess = rng.random() < positivity

        guess_is_correct = truth == guess

        evaluation[(truth, guess_is_correct)] += 1

    print(evaluation, f"Accuracy: {(evaluation[(False, True)] + evaluation[(True, True)])/len(dataset)}")


def main():

    example_naive()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()

