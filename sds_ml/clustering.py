import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import sds
import sds.variants
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
SILENT = 0

log = logging.getLogger(__name__)

def make_a_problem_space(lower, upper, sigma, dimensions, point_count, cluster_count, rng):

    points = []


    def assign_clusters():

        distributions = sorted(rng.sample(range(point_count), cluster_count-1)) + [point_count]

        current_cluster = 0

        log.log(0, 'distributions: %s', distributions)

        for point_num in range(point_count):
            
            log.log(0, "point num: %s, current cluster: %s", point_num, current_cluster)

            if point_num >= distributions[current_cluster]:

                current_cluster += 1

            log.log(0, "yielding %s", current_cluster)

            yield current_cluster

    point_clusters = list(assign_clusters())

    log.log(DEBUG, 'point clusters: %s', point_clusters)

    centroids = [
        [rng.uniform(lower, upper) for d in range(dimensions)]
        for cluster_num
        in range(cluster_count)
    ]

    points = [
        tuple(rng.gauss(x, sigma) for x in centroids[cluster_num])
        for cluster_num
        in point_clusters
    ]

    return points, point_clusters, centroids

def get_bounds(points):

    bounds = [
        [None, None]
        for dimension_num
        in range(len(points[0]))
    ]

    for point in points:

        log.log(0,"bounds %s", bounds)

        for feature, bound in zip(point, bounds):

            lower, upper = bound

            log.log(0, "feature: %s", feature)

            if lower is None or feature < lower:

                bound[0] = feature

            if upper is None or feature > upper:

                bound[1] = feature

    return bounds


        

def make_DH(points, dimension_count, max_k, rng):

    bounds = get_bounds(points)

    def DH():

        k = rng.randint(1, max_k)

        centroids = tuple(
            tuple(rng.uniform(*bounds[dimension_num]) for dimension_num in range(dimension_count))
            for centroid_num
            in range(k)
        )

        return centroids

    return DH

def microtest(hyp, point, threshold, dimension_count, rng):

    dimension = rng.randrange(dimension_count)

    feature = point[dimension]

    distances = (
        abs(centroid[dimension] - feature)
        for centroid
        in hyp
    )

    return any(distance < threshold for distance in distances)

def make_TM(points, dimension_count, threshold, rng):

    def TM():
        
        point = rng.choice(points)

        return functools.partial(
            microtest,
            point=point,
            threshold=threshold,
            dimension_count=dimension_count,
            rng=rng,
        )

    return TM

def noise(hyp, rng):

    return tuple(
        x+rng.gauss(0,0.001)
        for x
        in hyp
    )

def D_clustering(DH, swarm, rng):

    D_passive = sds.D_passive(DH, swarm, rng)

    def D(agent):

        polled = rng.choice(swarm)

        if agent.inactive and polled.active:

            agent.hyp = polled.hyp

        else:

            if agent.inactive or polled.active and len(agent.hyp) > len(polled.hyp):

                agent.active = False
                agent.hyp = DH()

    return D
    
def sds_clustering(points, max_k, dimension_count, threshold, H, swarm, rng):

    TM = make_TM(points=points, dimension_count=dimension_count, threshold=threshold, rng=rng)

    T = sds.T_boolean(TM)

    DH = make_DH(
        points=points,
        dimension_count=dimension_count,
        max_k=max_k,
        rng=rng,
    )

    D = D_clustering(DH, swarm, rng)

    I = sds.variants.I_async(D, T, swarm)

    sds.SDS(I=I, H=H)

def hyp_report(hyp):

    lines = []

    def centroid_report(centroid):

        nums = (format(x,".3f") for x in centroid)

        return f"[{', '.join(nums)}]"

    for num, centroid in enumerate(hyp):

        lines.append(
            f"centroid #{num}: {centroid_report(centroid)}"
        )

    return "\n".join(lines)

def cluster_report(swarm, min_cluster_proportion=0.2, max_clusters=5):

    return "\n".join(
        [
            f"Hypothesis #{hyp_num} has {len(hyp)} centroids.\n{hyp_report(hyp)}: {count}"
            for hyp_num, (hyp, count) in enumerate(
                swarm.clusters.most_common(max_clusters), start=1
            )
            if count > len(swarm) * min_cluster_proportion
        ]
    )


def main():

    # problem definition
    lower = 0
    upper = 1
    sigma = 0.01
    dimensions = 3
    point_count = 1000
    cluster_count = 3
    rng = random.Random(0)


    # solution definition
    min_iterations = 100
    max_iterations = 2000
    agent_count = 2000
    threshold_activity = 1
    max_k = cluster_count * 2
    threshold = sigma * 2


    points, point_clusters, centroids = make_a_problem_space(
        lower=lower,
        upper=upper,
        sigma=sigma,
        dimensions=dimensions,
        point_count=point_count,
        cluster_count=cluster_count,
        rng=rng,
    )

    log.info("Point cluster distribution: %s", sorted(collections.Counter(point_clusters).items()))

    log.info("Created problem with these centroids:\n%s", hyp_report(centroids))

    swarm = sds.Swarm(agent_count=agent_count)

    H = sds.variants.all_functions(
        sds.H_fixed(min_iterations),
        sds.variants.any_functions(
            sds.H_fixed(max_iterations),
            sds.variants.H_threshold(swarm, threshold_activity),
        )
    )

    log.info(
        "Running SDS for for [%s, %s] iterations with %s agents. Halting if activity reaches %.1f", 
        min_iterations,
        max_iterations,
        agent_count,
        threshold_activity,
    )


    sds_clustering(
        points=points,
        max_k=max_k,
        dimension_count=dimensions,
        threshold=threshold,
        swarm=swarm,
        H=H,
        rng=rng,
    )

    report = cluster_report(swarm)

    print(report)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()

