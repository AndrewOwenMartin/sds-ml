import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
import unittest
import sds_ml.clustering as clustering
import sds

log = logging.getLogger(__name__)


class TestClustering(unittest.TestCase):
    def setUp(self):

        logging.basicConfig(level=logging.DEBUG)

        self.log = logging.getLogger(__file__)

        self.rng = random.Random()

    @classmethod
    def setUpClass(cls):

        pass

    @classmethod
    def tearDownClass(cls):

        pass

    def test_make_problem_space(self):

        params = [
            dict(
                lower=0,
                upper=100,
                sigma=1,
                dimensions=3,
                point_count=50,
                cluster_count=4,
                rng=self.rng,
            ),
            dict(
                lower=0,
                upper=1,
                sigma=0.1,
                dimensions=14,
                point_count=5,
                cluster_count=2,
                rng=self.rng,
            ),
            dict(
                lower=0,
                upper=10,
                sigma=0.1,
                dimensions=2,
                point_count=10,
                cluster_count=3,
                rng=self.rng,
            ),
        ]

        for param_dict in params:

            points, point_clusters, centroids = clustering.problem.make_a_problem_space(
                **param_dict
            )

            log.info(
                "feature range: [%s-%s]. sigma=%3.1f",
                param_dict["lower"],
                param_dict["upper"],
                param_dict["sigma"],
            )

            for num, centroid in enumerate(centroids):

                log.info(
                    "centroid #%s: (%s)",
                    num,
                    ", ".join([format(x, ".2f") for x in centroid]),
                )

            for num, (point, cluster_num) in enumerate(zip(points, point_clusters)):

                log.info(
                    "point #%s, cluster %s: (%s)",
                    num,
                    cluster_num,
                    ", ".join([format(x, ".2f") for x in point]),
                )

    def test_get_bounds(self):

        points = [[-1, 0, 1], [0, 1, 0.5], [-0.5, 0.5, -1]]

        bounds = clustering.clustering.get_bounds(points)

        log.info("bounds %s", bounds)

    def test_make_DH(self):

        dimension_count = 3
        max_k = 5

        points = [[-1, -1, 0], [0, 1, 1]]

        DH = clustering.clustering.make_DH(
            points=points, dimension_count=dimension_count, max_k=max_k, rng=self.rng
        )

        for num in range(1, 11):

            hyp = DH()

            log.info(
                "DH #%s has %s centroids:\n%s",
                num,
                len(hyp),
                clustering.output.hyp_report(hyp),
            )

        try:
            d = {hyp: True}
        except TypeError:

            self.fail("hypothesis is not hashable")

    def test_microtest(self):

        point = [1, 0, 0]

        test_hyps = (
            ("bad", [[0.5, 0.5, 0], [0.2, 1, 0]]),
            ("good", [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        )

        for hyp_name, hyp in test_hyps:

            repeats = 10000

            results = (
                sum(
                    clustering.clustering.microtest(
                        hyp=hyp,
                        point=point,
                        dimension_count=len(point),
                        distance_metric=clustering.clustering.euclid_squared,
                        threshold=0.1,
                        rng=self.rng,
                    )
                    for repeat in range(repeats)
                )
                / repeats
            )

            log.info(
                "testing %s and %s %s. Result: %0.3f", point, hyp_name, hyp, results
            )

