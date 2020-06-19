import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
SILENT = 0

log = logging.getLogger(__name__)

def hyp_report(hyp):

    lines = []

    def centroid_report(centroid):

        nums = (format(x, ".3f") for x in centroid)

        return f"[{', '.join(nums)}]"

    for num, centroid in enumerate(hyp):

        lines.append(f"centroid #{num}: {centroid_report(centroid)}")

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
