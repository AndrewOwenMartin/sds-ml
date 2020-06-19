import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import argparse, sds
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
import sds_ml.clustering

SILENT = 0

log = logging.getLogger(__name__)


def get_bounds(points):
    """
    Given a list of n-dimensional points, returns a list of length n where each
    element is a tuple holding (min, max) for that dimension.

    points -- list of n-dimensional points
    """

    bounds = [[None, None] for dimension_num in range(len(points[0]))]

    for point in points:

        for feature, bound in zip(point, bounds):

            lower, upper = bound

            if lower is None or feature < lower:

                bound[0] = feature

            if upper is None or feature > upper:

                bound[1] = feature

    return bounds


def make_DH(points, dimension_count, max_k, rng=random):
    """
    Returns a function which has no arguments and returns a random hypothesis
    within the search space defined by the passed parameters.

    Positional arguments:
    points -- list of n-dimensional points
    dimension_count -- number of dimensions for each point
    max_k -- the maximum number of centroids to hypothesise

    Keyword arguments:
    rng -- an instance of random.Random (optional)
    """

    bounds = get_bounds(points)

    def DH():
        """
        Returns a hypothesis selected by generating a number of centroids
        between 1 and max_k (inclusive), where each centroid is uniformly
        distributed between the bounds of each dimension.
        """

        k = rng.randint(1, max_k)

        centroids = tuple(
            tuple(
                rng.uniform(*bounds[dimension_num])
                for dimension_num in range(dimension_count)
            )
            for centroid_num in range(k)
        )

        return centroids

    return DH


def microtest(hyp, point, dimension_count, distance_metric, threshold, rng=random):
    """
    Returns true if the hypothesis has any centroids within the threshold
    distance of the point as calculated by the given distance metric.

    Positional arguments:
    hyp -- a hypothesis
    point -- a point from the dataset
    dimension_count -- number of dimensions for each point
    distance_metric -- function for calculating distance
    threshold -- maximum acceptable distance

    Keyword arguments:
    rng -- an instance of random.Random (optional)
    """

    dimension = rng.randrange(dimension_count)

    feature = point[dimension]

    distances = (distance_metric(point, centroid) for centroid in hyp)

    return any(distance < threshold for distance in distances)


def make_boolean_TM(points, dimension_count, distance_metric, threshold, rng=random):
    """
    Returns a function which has no arguments and returns a random microtest.

    Positional arguments:
    points -- all points in the dataset
    dimension_count -- number of dimensions for each point
    distance_metric -- function for calculating distance
    threshold -- maximum acceptable distance

    Keyword arguments:
    rng -- an instance of random.Random (optional)
    """

    def TM():
        """
        Returns a microtest which partially evaluates a hypothesis.
        """

        point = rng.choice(points)

        return functools.partial(
            microtest,
            point=point,
            threshold=threshold,
            dimension_count=dimension_count,
            distance_metric=distance_metric,
            rng=rng,
        )

    return TM


def euclid_squared(vector_a, vector_b):

    return sum(abs(a - b) ** 2 for a, b in zip(vector_a, vector_b))


def example_basic():

    # problem definition
    lower = 0
    upper = 1
    sigma = 0.05
    dimensions = 3
    point_count = 100
    cluster_count = 4
    rng = random.Random()

    # solution definition
    max_iterations = 10 ** 3
    agent_count = 1000
    max_k = cluster_count * 2
    threshold = 0.1

    points, point_clusters, centroids = sds_ml.clustering.problem.make_a_problem_space(
        lower=lower,
        upper=upper,
        sigma=sigma,
        dimensions=dimensions,
        point_count=point_count,
        cluster_count=cluster_count,
        rng=rng,
    )

    log.info(
        "Point cluster distribution: %s",
        sorted(collections.Counter(point_clusters).items()),
    )

    log.info(
        "Created problem with these centroids:\n%s",
        sds_ml.clustering.output.hyp_report(centroids),
    )

    swarm = sds.Swarm(agent_count=agent_count)

    H = sds.H_fixed(max_iterations)

    log.info(
        "Running SDS for for %s iterations with %s agents.", max_iterations, agent_count
    )

    TM = make_boolean_TM(
        points=points,
        dimension_count=dimensions,
        distance_metric=euclid_squared,
        threshold=threshold,
        rng=rng,
    )

    T = sds.T_boolean(TM)

    DH = make_DH(points=points, dimension_count=dimensions, max_k=max_k, rng=rng)

    D = sds.D_passive(DH, swarm, rng)

    I = sds.I_sync(D, T, swarm)

    sds.SDS(I=I, H=H)

    report = sds_ml.clustering.output.cluster_report(swarm, min_cluster_proportion=0.01)

    log.info(report)


def main():
    parser = argparse.ArgumentParser(description="SDS clustering examples")

    parser.add_argument(
        "name", type=str, default="basic", help="Name of the example to run"
    )

    args = parser.parse_args()

    name2example = {"basic": example_basic}

    example = name2example[args.name]

    example()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()
