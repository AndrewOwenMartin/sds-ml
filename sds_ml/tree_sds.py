import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
import sds_ml.tree_search as tree_search
import sds
SILENT = 0

log = logging.getLogger(__name__)

def DH_tree(tree, rng):

    def DH(hyp):

        if hyp:

            return frozenset(rng.sample(hyp, len(hyp)-1))

        else:

            return frozenset()

def DD_tree(tree, split_num, rng):

    def DD(hyp):

        if hyp:

            mutable_hyp = list(hyp)

            random_index = rng.randrange(len(mutable_hyp))

            leaf_split = mutable_hyp.pop(random_index)



        else:

            leaf_split = tree


def count_leaf_splits(hyp):

    leaf_split_count = 0

    if all(isinstance(item, tree_search.Node) for item in hyp):

        return 1

    return sum(
        count_leaf_splits(item)
        for item
        in hyp
        if not isinstance(item, tree_search.Node)
    )

def prune(hyp, rng):

    leaf_split_count = count_leaf_splits(hyp)

    prune_index = rng.randrange(leaf_split_count)


    
class Hyp:

    def __init__(self, root_split):

        self.root_split = root_split

    def __str__(self):

        return str(self.root_split)

    def first_split(rng, tree, split_num):

        return Hyp(
            root_split=Split(
                members=sorted(
                    rng.sample(tree.children, split_num),
                    key=lambda node: node.index,
                )
            )
        )

    def random_split(self, tree, rng, split_num):

        if self.root_split:

            leaf_node_count = self.count_splittable_leaf_nodes()

            node_index = rng.randrange(leaf_node_count)

            clone_split = self.root_split.deep_clone()

            clone_split.split_node(node_num=node_index, split_num=split_num, rng=rng)

            return Hyp(root_split=clone_split)

        else:

            return Hyp.first_split(rng=rng, tree=tree, split_num=split_num)
            
    def random_prune(self, rng):

        leaf_split_count = self.count_leaf_splits()

        prune_split = rng.randrange(leaf_split_count)

        clone_split = self.root_split.deep_clone()

        log.debug("leaf_split_count: %s", leaf_split_count)

        clone_split.prune_leaf(leaf_num=prune_split)

        return Hyp(root_split=clone_split)

    def iter_split(self):

        pass

    def count_leaf_splits(self):

        if self.root_split:

            return self.root_split.count_leaf_splits()

        else:

            return 0

    def count_splittable_leaf_nodes(self):

        if self.root_split:

            return self.root_split.count_splittable_leaf_nodes()

        else:

            return 0


class Split:

    def __init__(self, members):

        self.members = members

    def is_leaf_split(self):

        return not any(isinstance(member,Split) for member in self.members)

    def prune(self):

        return self.members[0].parent

    def split_node(self, node_num, split_num, rng):

        node_count = 0

        for parent, member_num, member in self.bredth_first():

            if isinstance(member, Split):
                continue

            #log.info("parent %s, member_num: %s, member: %s, node_count: %s", parent, member_num, repr(member), node_count)

            if node_num == node_count:

                log.info("splitting node %s", member)

                parent.members[member_num] = Split(
                    members=sorted(
                        rng.sample(member.children, split_num),
                        key=lambda node: node.index,
                    )
                )

                return

            node_count += 1

            

    def bredth_first(self):

        queue = collections.deque([
            (self, num, member)
            for num, member
            in enumerate(self.members)
        ])

        while queue:

            parent, num, member = queue.popleft()

            yield (parent, num, member)

            if isinstance(member, Split):

                for child_num, child in enumerate(member.members):
                
                    queue.append(
                        (
                            member,
                            child_num,
                            child,
                        )
                    )
        

    def prune_leaf(self, leaf_num):

        if self.is_leaf_split() and leaf_num == 0:

            return None

        parent_num_member_queue = collections.deque([(self, member_num, member) for member_num, member in enumerate(self.members)])

        leaf_counter = 0

        while parent_num_member_queue:

            parent, num, member = parent_num_member_queue.popleft()

            if isinstance(member, Split):

                log.debug("found split: %s", member)
                
                if member.is_leaf_split():

                    log.debug("found leaf split #%s", leaf_counter)

                    if(leaf_counter == leaf_num):

                        log.debug("setting member %s of parent split to node %s. Current parent %s", num, member.members[0].parent.name, parent)

                        parent.members[num] = member.members[0].parent 

                        return
                    
                    else:

                        leaf_counter += 1

                        log.debug("leaf counter now %s", leaf_counter)

                log.debug("enqueueing members: %s", member.members)

                for child_num, child in enumerate(member.members):

                    parent_num_member_queue.append(
                        (
                            member,
                            child_num, 
                            child,
                        )
                    )

    def __repr__(self):

        return f"<Split: {len(self.members)}>"

    def count_splittable_leaf_nodes(self):

        leaf_node_count = 0

        for member in self.members:

            if isinstance(member, Split):

                leaf_node_count += member.count_splittable_leaf_nodes()

            else:

                if len(member.children) > 0:

                    log.info("node %s is a splittable leaf node", member)

                    leaf_node_count += 1

                else:

                    log.info("node %s is a leaf node, but it is not splittable, no children.", member)

        return leaf_node_count
                

    def count_leaf_splits(self):

        leaf_split_count = 0

        if all(isinstance(member, tree_search.Node) for member in self.members):

            return 1

        return sum(
            member.count_leaf_splits()
            for member
            in self.members
            if isinstance(member, Split)
        )

    def deep_clone(self):

        clone_members = []

        for member in self.members:

            if isinstance(member, Split):

                clone_members.append(member.deep_clone())

            else:

                clone_members.append(member)

        return Split(members=clone_members)

    def __str__(self):

        parts = []

        for member in self.members:

            if isinstance(member, tree_search.Node):

                part = member.name

            else:

                part = str(member)

            parts.append(part)

        return f"[{', '.join(parts)}]"

def clone_split(split, split_skip_delay=0):

    #if all(isinstance(item, tree_search.Node) for item in split):
    if all(isinstance(item, str) for item in split):

        if split_skip_delay == 0:

            return split[0][:-1]

        else:

            return split

    return [
        clone_split(item)
        for item
        in split
    ]
        
            

def D_tree_search(DH, DD, swarm, rng):

    def D(agent):

        if agent.inactive:

            polled = rng.choice(swarm)

            if polled.active:

                agent.hyp = DD(polled.hyp)

            else:

                agent.hyp = DH(agent.hyp)

            

    return D

def T_tree_search():

    def T(agent):

        pass

    return T

def I_tree(D, T, swarm):

    def I():

        for agent in swarm:

            backscatter(agent)

        for agent in swarm:

            T(agent)

    return I

            

        

def example(tree, swarm):

    D = None

    T = None

    I = sds.I_sync(D=D, T=T, swarm=swarm)

def main():

    tree = tree_search.build_tree(tree_depth=3, branch_count=3, split_num=2)

    swarm = sds.Swarm(agent_count=100)

    example(tree=tree, swarm=swarm)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
        style="%",
    )

    main()

