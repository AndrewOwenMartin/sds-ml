import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
import sds_ml
import sds_ml.sds_ml
import operator
SILENT = 0

log = logging.getLogger(__name__)

DimensionThreshold = collections.namedtuple("DimensionThreshold", ("dimension", "operator", "threshold"))

def DH_plane(dataset, rng):
    """ 
    Takes a "choice"able dataset where each row is expected to be of the same
    length.
    
    Each row must also be indexable and each value in a row is ordered with
    respect to all other values in the same column.

    Returns a single (feature_index, threshold, operator) tuple
    """

    def DH():

        random_row = rng.choice(dataset)

        feature_num = rng.randrange(len(random_row)-1)

        threshold = random_row[feature_num]

        operator = random.choice(sds_ml.sds_ml.operators)

        return DimensionThreshold(feature_num, operator, threshold)

    return DH

def TM_plane(dataset, rng):
    """
    Takes a (feature_index, threshold, operator) hyp and a row.

    Applies operator to row[feature_index] and compares result to row[-1]
    """

    def microtest(hyp, row):

        feature = row[hyp.dimension]

        comparison = hyp.operator(feature, hyp.threshold)

        return comparison == row[-1]

    def TM():

        row = rng.choice(dataset)

        return functools.partial(microtest, row=row)

    return TM, microtest

def DH_plane_union(dataset, rng):
    """
    Takes a "choice"able dataset where each row is expected to be of the same
    length.
    
    Each row must also be indexable and each value in a row is ordered with
    respect to all other values in the same column.

    Returns between one dimension threshold, or one dimension threshold per dimension, with no repeated dimensions.

    Returns an unordered set of (feature_index, threshold, operator) tuples
    """

    dimension_count = max(len(row)-1 for row in dataset)

    def DH():

        hyp_dim_count = rng.randint(2, dimension_count)

        hyp_dims = rng.sample(range(dimension_count), hyp_dim_count)

        hyp = frozenset(
            DimensionThreshold(
                dimension=dimension,
                operator=rng.choice(sds_ml.sds_ml.operators),
                threshold=rng.choice(dataset)[dimension],
            )
            for dimension
            in sorted(hyp_dims)
        )

        return hyp

    return DH

def TM_plane_intersection(dataset, rng):

    def microtest(hyp, row):

        comparison = all(
            plane.operator(row[plane.dimension], plane.threshold)
            for plane
            in hyp
        )

        return comparison == row[-1]

    def TM():

        row = rng.choice(dataset)

        return functools.partial(microtest, row=row)

    return TM, microtest

def TM_plane_union(dataset, rng):

    def microtest(hyp, row):

        comparison = any(
            plane.operator(row[plane.dimension], plane.threshold)
            for plane
            in hyp
        )

        return comparison == row[-1]

    def TM():

        row = rng.choice(dataset)

        return functools.partial(microtest, row=row)

    return TM, microtest

class Plane:

    __slots__ = ("dimension", "operator", "threshold")

    operator2symbol = {
        operator.lt:"<",
        operator.gt:">",
    }

    operators = tuple(operator2symbol)

    def __init__(self, dimension, operator, threshold):

        self.dimension = dimension
        self.operator = operator
        self.threshold = threshold

    def __eq__(self, other):

        return self.dimension == other.dimension and self.operator is other.operator

    def __call__(self, row):

        return self.operator(row[self.dimension], self.threshold)

    def __hash__(self):

        return hash((self.dimension, self.operator))

    def __str__(self):

        return f"{self.dimension} {Plane.operator2symbol[self.operator]} {self.threshold:.2g}"

class IndexSet(collections.UserList):

    __slots__ = ("data","signature")

    def __init__(self, elements):

        self.data = tuple(elements)

        self.signature = frozenset(elements)

    def __eq__(self, other):

        return other.signature == self.signature

    def __lt__(self, other):

        return other.signature.contains(self.signature)

    def __gt__(self, other):

        return self.signature.contains(other.signature)

    def __hash__(self):

        return hash(self.signature)


def DH_data_driven(dataset, swarm, rng):

    dimension_count = max(len(row)-1 for row in dataset)

    def select_intersection_dim_count():

        polled = random.choice(swarm)

        if polled.active:

            intersection = rng.choice(polled.hyp)

            return len(intersection)

        else:

            return rng.randint(1, dimension_count)

    def select_intersection_count():

        polled = random.choice(swarm)

        if polled.active:

            return max(1, len(polled.hyp) + round(rng.gauss(0,1)))

        else:

            return 1 + abs(round(rng.gauss(0,1)))

    def select_dimension():

        polled = random.choice(swarm)

        if polled.active:

            intersection = rng.choice(polled.hyp)

            plane = rng.choice(intersection)

            return plane.dimension

        else:

            return rng.randrange(1, dimension_count)

    def select_operator():

        polled = random.choice(swarm)

        if polled.active:

            intersection = rng.choice(polled.hyp)

            plane = rng.choice(intersection)

            return plane.operator

        else:

            return rng.choice(Plane.operators)

    def select_threshold(dimension):

        polled = random.choice(swarm)

        if polled.active:

            intersection = rng.choice(polled.hyp)

            plane = rng.choice(intersection)

            if dimension == plane.dimension:

                return plane.threshold

        return rng.choice(dataset)[dimension]
            

    def select_plane():

        dimension = select_dimension()

        operator = select_operator()

        threshold = select_threshold(dimension)

        return Plane(
            dimension=dimension,
            operator=operator,
            threshold=threshold,
        )

    def DH():

        intersections = []

        for intersection_num in range(select_intersection_count()):

            intersection = IndexSet(
                tuple(
                    select_plane()
                    for dim_num
                    in range(select_intersection_dim_count())
                )
            )

            intersections.append(intersection)

        union = IndexSet(intersections)

        return union

    return DH


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

