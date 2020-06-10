import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import unittest
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
import sds_ml.tree_sds as tree_sds
import sds_ml.tree_search as tree_search

log = logging.getLogger(__name__)

class TestTreeSds(unittest.TestCase):

    def setUp(self):

        logging.basicConfig(level=logging.DEBUG)

        self.log = logging.getLogger(__file__)

        self.rng = random.Random()

        self.tree = tree_search.build_tree(depth=5, branch_count=10)

        self.name2node = {
            node.name: node
            for node
            in self.tree.bredth_first()
        }

    @classmethod
    def setUpClass(cls):

        pass

    @classmethod
    def tearDownClass(cls):

        pass

    def test_name2node(self):

        log.log(
            INFO,
            "\n%s",
            "\n".join(
                itertools.islice(
                    (str(x) for x in self.name2node.items()),
                    50,
                )
            )
        )

    def test_build_hyp(self):

        hyp0 = tree_sds.Hyp(
            root_split=None,
        )

        log.info("hyp0: %s", hyp0)

        log.info("hyp0 leaf splits: %s", hyp0.count_leaf_splits())

        log.info("hyp0 splittable leaf nodes: %s", hyp0.count_splittable_leaf_nodes())

        hyp_simple = tree_sds.Hyp(
            root_split=tree_sds.Split(
                members=[
                    self.name2node["r0"],
                    self.name2node["r1"],
                    self.name2node["r2"],
                    self.name2node["r3"],
                ]
            )
        )

        log.info("hyp simple: %s", hyp_simple)

        log.info("hyp simple leaf splits: %s", hyp_simple.count_leaf_splits())

        log.info("hyp simple splittable leaf nodes: %s", hyp_simple.count_splittable_leaf_nodes())

        hyp1 = tree_sds.Hyp(
            root_split=tree_sds.Split(
                members=[
                    tree_sds.Split(
                        members=[
                            self.name2node["r00"],
                            tree_sds.Split(
                                members=[
                                    self.name2node["r010"],
                                    self.name2node["r011"],
                                    tree_sds.Split(
                                        members=[
                                            self.name2node["r0120"],
                                            self.name2node["r0121"],
                                            self.name2node["r0122"],
                                            self.name2node["r0123"],
                                        ]
                                    ),
                                    self.name2node["r013"],
                                ]
                            ),
                            self.name2node["r02"],
                            self.name2node["r03"],
                        ]
                    ),
                    self.name2node["r1"],
                    self.name2node["r2"],
                    tree_sds.Split(
                        members=[
                            self.name2node["r30"],
                            self.name2node["r31"],
                            self.name2node["r32"],
                            self.name2node["r33"],
                        ]
                    ),
                ]
            )
        )

        log.info("hyp1: %s", hyp1)

        log.info("hyp1 leaf splits: %s", hyp1.count_leaf_splits())

        log.info("hyp1 splittable leaf nodes: %s", hyp1.count_splittable_leaf_nodes())

        hyp2 = tree_sds.Hyp(
            root_split=tree_sds.Split(
                members=[
                    tree_sds.Split(
                        members=[
                            self.name2node["r00"],
                            self.name2node["r01"],
                            self.name2node["r02"],
                            self.name2node["r03"],
                        ]
                    ),
                    tree_sds.Split(
                        members=[
                            self.name2node["r10"],
                            self.name2node["r11"],
                            self.name2node["r12"],
                            self.name2node["r13"],
                        ]
                    ),
                    tree_sds.Split(
                        members=[
                            self.name2node["r20"],
                            self.name2node["r21"],
                            self.name2node["r22"],
                            self.name2node["r23"],
                        ]
                    ),
                    tree_sds.Split(
                        members=[
                            self.name2node["r30"],
                            self.name2node["r31"],
                            self.name2node["r32"],
                            self.name2node["r33"],
                        ]
                    ),
                ]
            )
            
        )

        log.info("hyp2: %s", hyp2)

        log.info("hyp2 leaf splits: %s", hyp2.count_leaf_splits())

        log.info("hyp2 splittable leaf nodes: %s", hyp2.count_splittable_leaf_nodes())

        root_hyp = tree_sds.Hyp.first_split(rng=self.rng, tree=self.tree, split_num=4)

        log.info("root hyp: %s", root_hyp)

        log.info("root hyp leaf splits: %s", root_hyp.count_leaf_splits())

        log.info("root hyp splittable leaf nodes: %s", root_hyp.count_splittable_leaf_nodes())

        for leaf_num in range(hyp1.count_leaf_splits()):

            clone_split = hyp1.root_split.deep_clone()

            clone_split.prune_leaf(leaf_num=leaf_num)

            prune_hyp = tree_sds.Hyp(root_split=clone_split)

            log.info("prune_hyp%s: %s", leaf_num, prune_hyp)

            log.info("prune_hyp%s leaf splits: %s", leaf_num, prune_hyp.count_leaf_splits())

            log.info("prune_hyp%s splittable leaf nodes: %s", leaf_num, prune_hyp.count_splittable_leaf_nodes())

        log.info("counting hyp1 splittable leaf nodes")

        for node_num in range(hyp1.count_splittable_leaf_nodes()):

            #log.info("hyp1 leaf node %s", node_num)

            hyp1_clone = hyp1.root_split.deep_clone()

            hyp1_clone.split_node(node_num=node_num, split_num=4, rng=self.rng)

            grow_hyp = tree_sds.Hyp(root_split=hyp1_clone)

            log.info("grow hyp1 node %s: %s", node_num, grow_hyp)






        #hyp2 = tree_sds.Hyp(splits=[])

        #hyp3 = hyp2.random_split(tree=self.tree, split_num=3, rng=self.rng)

        #log.info("%s", hyp3)

        #leaf_split_count = tree_sds.count_leaf_splits
