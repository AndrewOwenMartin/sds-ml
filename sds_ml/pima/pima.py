import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import sds_ml
import sds_ml.sds_ml
import sds_ml.pima
import sds_ml.variants
import sds
import sds.variants
import operator
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
SILENT = 0

log = logging.getLogger(__name__)

def data_driven_precision_and_recall(hyp, dataset, microtest):

    results = collections.Counter([
        (
            row[-1],
            microtest(hyp, row)
        )
        for row
        in dataset
    ])

    accuracy = (results[(1,1)] + results[(0,1)]) / sum(results.values())

    return dict(
        true_positive=results[(1,1)],
        true_negative=results[(0,1)],
        false_positive=results[(0,0)],
        false_negative=results[(1,0)],
        precision=results[(1,1)]/max(1, (results[(1,1)]+results[(0,0)])),
        recall=results[(1,1)]/max(1, (results[(1,1)]+results[(1,0)])),
        accuracy=accuracy,
    )

def example_data_driven_pima():

    rng = random.Random()

    dataset = sds_ml.pima.load()

    fields = next(iter(dataset))._fields

    agent_count = 10000

    max_iterations = 100

    swarm = sds.Swarm(agent_count=agent_count)

    DH = sds_ml.variants.DH_data_driven(dataset=dataset, swarm=swarm, rng=rng)

    D = sds.variants.D_context_sensitive(DH=DH, swarm=swarm, rng=rng)

    def microtest(hyp, row):

        return any(
            all(
                plane(row)
                for plane
                in intersection
            )
            for intersection
            in hyp
        ) == row[-1]

    TM = sds.TM_uniform(
        [
            functools.partial(
                microtest,
                row=row,
            )
            for row
            in dataset
        ],
        rng=rng,
    )

    T = sds.T_boolean(TM=TM)

    def union_to_str(union):

        x = " OR ".join(intersection_to_str(intersection) for intersection in union)

        return x
    

    def intersection_to_str(intersection):

        x = " AND ".join(plane_to_str(plane) for plane in intersection)

        return f"({x})"

    def plane_to_str(plane):

        return f"{fields[plane.dimension][:8]} {sds_ml.sds_ml.operator2symbol[plane.operator]} {plane.threshold:5.5g}"

    def report(iterations):

        clusters = swarm.clusters

        top_clusters = ", ".join(
            [
                f"({union_to_str(hyp)}, {size})"
                for (
                    hyp,
                    size,
                ) in clusters.most_common(3)
            ]
        )

        log.info(
            "%6s: %s",
            iterations,
            top_clusters,
        )

    I = sds.I_sync(D=D, T=T, swarm=swarm)
    I = sds.variants.I_report(
        I=I,
        report_num=1000,
        report_function=report,
    )
    #H = sds.H_fixed(iterations=max_iterations)
    def H():
        return False

    try:

        sds.SDS(I=I, H=H)

    except KeyboardInterrupt:

        pass

    cluster = swarm.largest_cluster

    X = data_driven_precision_and_recall(cluster.hyp, dataset, microtest)

    log.info(X)

    log.info("Hyp: %s, size: %0.3f, evaluate: %s", union_to_str(cluster.hyp), cluster.size, X)


def D_dimension_operator_sensitive(DH, swarm, rng):

    def contains_dimensions_and_operators(agent, polled):

        agent_dimensions_and_operators = set((component.dimension, component.operator) for component in agent)

        polled_dimensions_and_operators = set((component.dimension, component.operator) for component in polled)

        return agent_dimensions_and_operators.issuperset(polled_dimensions_and_operators)

    def D(agent):

        polled = rng.choice(swarm)

        if polled.active and agent.inactive:

            agent.hyp = polled.hyp

        else:

            if agent.inactive or polled.active and contains_dimensions_and_operators(agent.hyp, polled.hyp):

                agent.active = False

                agent.hyp = DH()

    return D

def example_plane_union_intersection_pima(set_type):

    rng = random.Random()

    dataset = sds_ml.pima.load()

    fields = next(iter(dataset))._fields

    agent_count = 10000

    max_iterations = 5000

    DH = sds_ml.variants.DH_plane_union(dataset=dataset, rng=rng)

    if set_type == "union":

        TM, microtest = sds_ml.variants.TM_plane_union(dataset=dataset, rng=rng)

    else:

        TM, microtest = sds_ml.variants.TM_plane_intersection(dataset=dataset, rng=rng)

    swarm = sds.Swarm(agent_count=agent_count)

    #D = sds.D_passive(DH=DH, swarm=swarm, rng=rng)
    #D = sds.variants.D_context_free(DH=DH, swarm=swarm, rng=rng) # no convergence
    D = D_dimension_operator_sensitive(DH=DH, swarm=swarm, rng=rng)

    T = sds.T_boolean(TM=TM)

    def hyp_to_str(hyp):

        if set_type == "union":

            s = " OR "

        else:

            s = " AND "

        return s.join([
            f"({fields[dim_thr.dimension]} {sds_ml.sds_ml.operator2symbol[dim_thr.operator]} {dim_thr.threshold:.2f})"
            for dim_thr
            in hyp
        ])

    def report(iterations):

        clusters = swarm.clusters

        top_clusters = ", ".join(
            [
                f"({hyp_to_str(hyp)}, {size})"
                for (
                    hyp,
                    size,
                ) in clusters.most_common(3)
            ]
        )

        log.info(
            "%6s * %5s = %8s: %s",
            iterations,
            agent_count,
            iterations * agent_count,
            top_clusters,
        )

    I = sds.I_sync(D=D, T=T, swarm=swarm)
    I = sds.variants.I_report(
        I=I,
        report_num=200,
        report_function=report,
    )
    H = sds.H_fixed(iterations=max_iterations)

    sds.SDS(I=I, H=H)

    cluster = swarm.largest_cluster

    log.info("Hyp: %s, size: %0.3f, evaluate: %.2f%%", hyp_to_str(cluster.hyp), cluster.size, evaluate(cluster.hyp, microtest, dataset)*100)

    log.info("cluster %s", cluster)

    if set_type == "union":

        p_and_r = plane_union_precision_and_recall(cluster.hyp, dataset)

    else:

        p_and_r = plane_intersection_precision_and_recall(cluster.hyp, dataset)

    log.info("precision and recall: %s", p_and_r)


def evaluate(hyp, microtest, dataset):

    correct = sum(microtest(hyp, row) for row in dataset)

    return correct/len(dataset)

def plane_intersection_precision_and_recall(hyp, dataset):

    results = collections.Counter([
        (
            row[-1],
            all(
                hyp_component.operator(row[hyp_component.dimension], hyp_component.threshold)
                for hyp_component
                in hyp
            )
        )
        for row
        in dataset
    ])

    return dict(
        true_positive=results[(1,1)],
        true_negative=results[(0,0)],
        false_positive=results[(0,1)],
        false_negative=results[(1,0)],
        precision=results[(1,1)]/(results[(1,1)]+results[(0,1)]),
        recall=results[(1,1)]/(results[(1,1)]+results[(1,0)]),
    )

def plane_union_precision_and_recall(hyp, dataset):

    results = collections.Counter([
        (
            row[-1],
            any(
                hyp_component.operator(row[hyp_component.dimension], hyp_component.threshold)
                for hyp_component
                in hyp
            )
        )
        for row
        in dataset
    ])

    return dict(
        true_positive=results[(1,1)],
        true_negative=results[(0,0)],
        false_positive=results[(0,1)],
        false_negative=results[(1,0)],
        precision=results[(1,1)]/(results[(1,1)]+results[(0,1)]),
        recall=results[(1,1)]/(results[(1,1)]+results[(1,0)]),
    )

def plane_precision_and_recall(hyp, dataset):

    results = collections.Counter([
        (
            row[-1],
            hyp.operator(row[hyp.dimension], hyp.threshold),
        )
        for row
        in dataset
    ])

    return dict(
        true_positive=results[(1,1)],
        true_negative=results[(0,0)],
        false_positive=results[(0,1)],
        false_negative=results[(1,0)],
        precision=results[(1,1)]/(results[(1,1)]+results[(0,1)]),
        recall=results[(1,1)]/(results[(1,1)]+results[(1,0)]),
    )


def example_threshold_pima():
    """ thresholding against a single dimension with the PIMA dataset"""

    rng = random.Random()

    dataset = sds_ml.pima.load()

    fields = next(iter(dataset))._fields

    agent_count = 1000

    max_iterations = 2000

    DH = sds_ml.variants.DH_plane(dataset=dataset, rng=rng)

    TM, microtest = sds_ml.variants.TM_plane(dataset=dataset, rng=rng)

    swarm = sds.Swarm(agent_count=agent_count)

    D = sds.D_passive(DH=DH, swarm=swarm, rng=rng)
    #D = sds.variants.D_context_free(DH=DH, swarm=swarm, rng=rng)
    # D = context_sensitive_sds.D_context_sensitive(DH=DH, swarm=swarm, rng=rng)

    T = sds.T_boolean(TM=TM)

    def report(iterations):

        clusters = swarm.clusters

        top_clusters = ", ".join(
            [
                f"({fields[hyp.dimension]} {sds_ml.sds_ml.operator2symbol[hyp.operator]} {hyp.threshold:.2f}, {size})"
                for (
                    hyp,
                    size,
                ) in clusters.most_common(3)
            ]
        )

        log.info(
            "%6s * %5s = %8s: %s",
            iterations,
            agent_count,
            iterations * agent_count,
            top_clusters,
        )

    I = sds.I_sync(D=D, T=T, swarm=swarm)
    I = sds.variants.I_report(
        I=I,
        report_num=100,
        report_function=report,
    )
    H = sds.H_fixed(iterations=max_iterations)

    sds.SDS(I=I, H=H)

    cluster = swarm.largest_cluster

    log.info("cluster: %s, evaluate: %.2f%%", cluster, evaluate(cluster.hyp, microtest, dataset)*100)

def main():

    example_threshold_pima()
    example_plane_union_pima()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()

