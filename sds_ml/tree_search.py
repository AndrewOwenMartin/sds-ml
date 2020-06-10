import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
import sds
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
SILENT = 0

log = logging.getLogger(__name__)


class Node:

    def __init__(self, parent, name, index, child_num, data_f):

        self.name = name
        self.index = index
        self.parent = parent
        self.children = []
        self.child_num = child_num
        self.data = data_f(self)

    def __str__(self):

        return f"{self.name}: {self.data}"

    def __repr__(self):

        return f"<Node {self.name}>"

    def make_children(self, depth, branch_count, data_f):

        if depth <= 0:

            return

        for child_num in range(branch_count):

            child_name = f"{self.name}{child_num}"

            child = Node(name=child_name, child_num=child_num, parent=self, data_f=data_f, index=self.index + (child_num,))

            self.children.append(child)

            log.log(0, "recursing to depth %s for child#%s '%s'", depth-1, child_num, child_name)

            child.make_children(depth=depth-1, branch_count=branch_count, data_f=data_f)

    def report(self):

        return f"Node '{self.name}' has this parent '{self.parent and self.parent.name or None}' and these {len(self.children)} children [{', '.join(child.name for child in self.children)}]"

    def iter_children(self):

        yield from iter(self.children)

    def bredth_first(self):

        queue = collections.deque([self])

        log.log(0, "bredth first search from %s", self.name)

        while queue:

            node = queue.popleft()

            log.log(0, "yielding %s. Data: %s", node.name, node.data)

            yield node

            for child in node.iter_children():

                queue.append(child)

    def depth_first(self):

        log.log(0, "yielding %s. Data: %s", self.name, self.data)

        yield self

        for child in self.iter_children():

            yield from child.depth_first()

    def shape_report(self):

        indices = [node.name for node in self.bredth_first()]

        return f"[{' '.join(indices)}]"

            
def build_tree(depth, branch_count, data_f=None):

    if data_f is None:

        data_f = lambda node: None
    
    root = Node(parent=None, name="r", data_f=data_f, child_num=0, index=("root",))

    log.log(0, "starting at depth=%s, branch_count=%s", depth, branch_count)

    root.make_children(depth=depth-1, branch_count=branch_count, data_f=data_f)

    return root

def make_search_space(depth, branch_count, rng):

    def random_data(node):

        return rng.random()

    root = build_tree(depth=depth, branch_count=branch_count, data_f=random_data)

    name2node = {
        node.index: node
        for node
        in root.bredth_first()
    }

    for name, node in name2node.items():

        log.info("%s: %s", name, node.data)

#def get_binary_subtrees(node):
#
#    yield from get_subtrees(node, split_num=2)

def get_binary_subtrees(node):

    yield (node,)

    log.log(0, "yielding the number of ways to pick two from the node")

    child_pairs = itertools.combinations(node.children, 2)

    for a, b in child_pairs:

        a_subtrees = get_binary_subtrees(a)

        b_subtrees = get_binary_subtrees(b)

        for num, subtree_combinations in enumerate(itertools.product(a_subtrees, b_subtrees)):

            yield list(itertools.chain.from_iterable(subtree_combinations))

def get_subtrees(node, split_num):

    yield (node,)

    # child combinations is a generator of node tuples. Each tuple is of length split_num.
    child_combinations = itertools.combinations(node.children, split_num)

    for child_combination in child_combinations:

        # child combination is a tuple of nodes of length split_num
        subtree_iterators = (
            get_subtrees(node, split_num=split_num)
            for node
            in child_combination
        )

        for num, subtree_combinations in enumerate(itertools.product(*subtree_iterators)):

            yield list(itertools.chain.from_iterable(subtree_combinations))


def main():

    rng = random.Random()

    make_search_space(3, 3, rng=rng)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()

