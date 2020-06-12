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

    def test_make_DH(self):

        dimension_count = 3
        max_k = 5

        points = [[-1, -1, 0], [0, 1, 1]]

        DH = clustering.make_DH(
            points=points, dimension_count=dimension_count, max_k=max_k, rng=self.rng
        )

        for num in range(1, 11):

            hyp = DH()

            log.info(
                "DH #%s has %s centroids:\n%s",
                num,
                len(hyp),
                clustering.hyp_report(hyp),
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
                    clustering.microtest(
                        point=point,
                        hyp=hyp,
                        threshold=0.1,
                        dimension_count=len(point),
                        rng=self.rng,
                    )
                    for repeat in range(repeats)
                )
                / repeats
            )

            log.info(
                "testing %s and %s %s. Result: %0.3f", point, hyp_name, hyp, results
            )

    def test_make_TM(self):

        TM = clustering.make_TM(
            points=[[0, 0, 0], [0, 1, 0]],
            dimension_count=3,
            threshold=0.1,
            rng=self.rng,
        )

        microtest = TM()

        result = microtest(hyp=[[0, 0, 0]])

        log.info("test make TM: %s", result)

    def test_get_bounds(self):

        points = [[-1, 0, 1], [0, 1, 0.5], [-0.5, 0.5, -1]]

        bounds = clustering.get_bounds(points)

        log.info("bounds %s", bounds)

    def test_make_problem_space(self):

        params = [
            dict(lower=0, upper=100, sigma=1, dimensions=3, point_count=50, cluster_count=4, rng=self.rng),
            dict(lower=0, upper=1, sigma=0.1, dimensions=14, point_count=5, cluster_count=2, rng=self.rng),
            dict(lower=0, upper=10, sigma=0.1, dimensions=2, point_count=10, cluster_count=3, rng=self.rng),
        ]

        for param_dict in params:

            points, point_clusters, centroids = clustering.make_a_problem_space(
                **param_dict
            )

            log.info("feature range: [%s-%s]. sigma=%3.1f", param_dict["lower"], param_dict["upper"], param_dict["sigma"])

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

    def test_simple_sds_clustering(self):

        points = [[0], [0.5], [1], [0.49]]

        max_k = 5

        if len(set(len(point) for point in points)) > 1:

            raise ValueError("not all same length points")

        dimension_count = len(points[0])

        threshold = 0.05

        agent_count = 500

        repeats = 3

        for repeat_num in range(repeats):

            H = sds.H_fixed(200 * repeat_num + 200)

            swarm = sds.Swarm(agent_count=agent_count)

            clustering.sds_clustering(
                points=points,
                max_k=max_k,
                dimension_count=dimension_count,
                threshold=threshold,
                swarm=swarm,
                H=H,
                rng=self.rng,
            )

            cluster_report = clustering.cluster_report(swarm)

            log.info("Repeat #%s.\n%s", repeat_num, cluster_report)

            log.info("Activity %s", swarm.activity)

            log.info(
                "hyp lengths %s",
                collections.Counter(len(agent.hyp) for agent in swarm).most_common(),
            )
