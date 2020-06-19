import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
from logging import DEBUG, INFO, WARNING, ERROR, FATAL

SILENT = 0

log = logging.getLogger(__name__)


def make_a_problem_space(
    lower, upper, sigma, dimensions, point_count, cluster_count, rng=random.Random()
):
    """
    Make a problem space for clustering and the ground truth 'answers'.
    
    Keyword arguments:
    lower -- lower bound for a value in any dimension
    upper -- upper bound for a value in any dimension
    dimensions -- the number of dimensions in each point
    point_count -- the total number of points to generate
    cluster_count -- the number of distinct clusters to generate
    rng -- an instance of random.Random (optional)
    """

    points = []

    def assign_clusters():

        distributions = sorted(rng.sample(range(point_count), cluster_count - 1)) + [
            point_count
        ]

        current_cluster = 0

        log.log(0, "distributions: %s", distributions)

        for point_num in range(point_count):

            log.log(0, "point num: %s, current cluster: %s", point_num, current_cluster)

            if point_num >= distributions[current_cluster]:

                current_cluster += 1

            log.log(0, "yielding %s", current_cluster)

            yield current_cluster

    point_clusters = list(assign_clusters())

    log.log(DEBUG, "point clusters: %s", point_clusters)

    centroids = [
        [rng.uniform(lower, upper) for d in range(dimensions)]
        for cluster_num in range(cluster_count)
    ]

    points = [
        tuple(rng.gauss(x, sigma) for x in centroids[cluster_num])
        for cluster_num in point_clusters
    ]

    return points, point_clusters, centroids
