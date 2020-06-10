import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import unittest
import heapq
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
import sds_ml.tree_search as tree_search

log = logging.getLogger(__name__)

class TestTreeSearch(unittest.TestCase):

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

    def test_make_tree(self):

        sml_tree = tree_search.build_tree(depth=2, branch_count=2, data_f=lambda node: None)

        log.info("sml_tree %s", sml_tree.report())

        lrg_tree = tree_search.build_tree(depth=3, branch_count=3, data_f=lambda node: node.parent and (node.parent.data * 2 + node.child_num) or 1)

        log.info("lrg_tree %s", lrg_tree.report())

        for node in sml_tree.bredth_first():

            #log.info(node.name)
            pass

        log.log(DEBUG, "depth first search from %s", sml_tree.name)

        for node in sml_tree.depth_first():

            #log.info(node.name)
            pass

    def test_get_binary_subtrees(self):

        tree = tree_search.build_tree(depth=3, branch_count=3)

        binary_subtrees = list(tree_search.get_binary_subtrees(tree))

        log.info("subtrees of tree: %s", tree.shape_report())

        expected_depth_3_branch_count_3_binary_subtrees = 49

        self.assertEqual(
            sum(1 for x in binary_subtrees),
            expected_depth_3_branch_count_3_binary_subtrees,
        )

        for num, subtree in enumerate(binary_subtrees):

            log.info("#%3s: %s", num, ", ".join(node.name for node in subtree))

        binary_subtrees = list(tree_search.get_subtrees(tree, split_num=2))

        self.assertEqual(
            sum(1 for x in binary_subtrees),
            expected_depth_3_branch_count_3_binary_subtrees,
        )

        for num, subtree in enumerate(binary_subtrees):

            log.info("#%3s: %s", num, ", ".join(node.name for node in subtree))

    def test_get_subtrees(self):

        tree = tree_search.build_tree(depth=3, branch_count=4)

        expected_depth_3_branch_count_4_split_3_subtrees = 501

        subtrees = tree_search.get_subtrees(tree, split_num=3)

        for num, subtree in enumerate(subtrees, start=1):

            log.info("#%3s: %s", num, ", ".join(node.name for node in subtree))

        self.assertEqual(num, expected_depth_3_branch_count_4_split_3_subtrees)
            
    def test_best_subtree(self):

        self.rng.seed(1)

        def node_num(node):

            return self.rng.random()

        tree = tree_search.build_tree(depth=3, branch_count=4, data_f=node_num)

        for node in tree.bredth_first():

            log.info("%s: %.3f", node.name, node.data)

        subtree_lists = (list(subtree) for subtree in tree_search.get_subtrees(tree, split_num=3))

        subtree_scores = [
            (
                " ".join(node.name for node in subtree),
                sum(node.data for node in subtree)/len(subtree),
            )
            for subtree
            in subtree_lists
        ]

        subtree_scores.sort(key=lambda x:x[1], reverse=True)

        log.info("scores")

        for name, score in subtree_scores[:20]:

            log.info("[%s] %.3f", name, score)

    def test_all_best_subtrees(self):

        tree_depths = [1,2,3,4,5]

        branch_counts = [1,2,3,4,5]

        split_num = [1,2,3,4,5]

        # tree_depths = itertools.count(start=1)

        # branch_counts = itertools.count(start=1)

        # split_num = itertools.count(start=1)

        def node_num(node):

            return self.rng.random()

        def subtree_score(leaf_list):

            return sum(node.data for node in leaf_list)/len(leaf_list)

        subtree_count = 0

        def iter_counter(i):

            nonlocal subtree_count

            subtree_count = 0

            for item in i:

                subtree_count += 1                

                yield item

        for tree_depth, branch_count, split_num in itertools.product(tree_depths, branch_counts, split_num):

            log.info("building tree. tree depth: %s, branch count: %s", tree_depth, branch_count)

            tree = tree_search.build_tree(depth=tree_depth, branch_count=branch_count, data_f=node_num)

            log.info("getting subtrees of split: %s", split_num)

            subtree_gen = iter_counter(tree_search.get_subtrees(tree, split_num=split_num))

            best_tree = max(subtree_gen, key=subtree_score)

            log.info("Evaluated %s subtrees", subtree_count)

            log.info("best tree has score %0.3f. [%s]\n---", subtree_score(best_tree), " ".join(node.name for node in best_tree))


            
            

        

    def test_foo(self):

        six = 6

        expected = 6

        self.assertEqual(
            six,
            expected,
            "Expected %s to equal %s" % (six, expected)
        )
