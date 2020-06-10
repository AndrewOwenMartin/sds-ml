import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import sds
import operator
import sds_ml.examples.pima
import sds
import sds.variants
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
SILENT = 0

log = logging.getLogger(__name__)

operator2symbol = {
    operator.lt:"<",
    operator.gt:">",
}

operators = (operator.lt, operator.gt)

def plane_union_hyp_to_str(hyp):


    return " OR ".join(
        [
            f"X[{hyp_component.dimension}] {operators[hyp_component.operator]} {hyp_component.threshold:g}"
            for hyp_component
            in hyp
        ]
    )




def example_xor():
    """ xor dataset against the plane-union method of hypothesis selection """

    rng = random.Random()

    xor_dataset = (
        (0,0),
        (0,1),
        (1,0),
        (1,1),
    )

    DH = DH_plane_union(dataset=xor_dataset, rng=rng)

    for num in range(10):

        hyp = DH()

        print(num, plane_union_hyp_to_str(hyp))



def main():

    sds_ml.examples.pima.example_data_driven_pima()
    #sds_ml.examples.pima.example_plane_union_intersection_pima(set_type="intersection")
    #sds_ml.examples.pima.example_threshold_pima()






if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()

