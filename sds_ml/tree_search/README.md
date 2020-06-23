# SDS Subtree Search

This is a record of my progress with using SDS to locate optimal subtrees which is a subtask of using SDS to fit a hyperplane to a set of data.

## SDS for game tree search.

This work takes existing work on Game Tree evaluation as a starting point. J. Mark Bishop and Thomas Tanay gave a presentation on [using SDS for game tree search](https://core.ac.uk/download/pdf/131208863.pdf). In this presentation a variant of SDS was described with these essential properties.

* **Initialisation** Create a population of agents at the root node of a game tree.
* **Initial Test** Each agent in the root node hypothesises a subsequent node and traverses the tree from that node to a leaf node using randomly selected child nodes. If the leaf node represents a victory for the player the agent becomes active.
* **Diffusion** Consists of three phases.
  * **Backscattering** All agents that were not contacted to form a test-hypothesis return to their parent node population.
  * **Scattering** Every active agent contacts another agent in the local population at random. If the contacted agent is inactive, it is sent onwards to the node indicated by the hypothesis of the polling agent.
  * **Internal Diffusion** Standard passive diffusion of hypotheses for agents within their local population.

## Problem redefinition - Optimal Binary Subtree search

To avoid complications arising from the specifics of hypergeometry I decided to first tacke an analogous problem, called optimal binary subtree search. In this problem you assume a large tree with many layers and many children for each node, each node holds a value between 0 and 1. The task is to determine which binary subtree has the greatest average value at its leaf nodes.

Please contact me if you feel this is not a suitable parallel problem to work on before tackling hyperplane fitting.

For this to work I first had to describe a tree, which I did using Python.

I defined in module `tree_search.py` a `Node` class which describes a tree of arbirary size and depth and which holds data at each node. I also wrote some functions for building and traversing such trees. I also wrote a simple comprehensive search function which iterates over all possible subtrees and returns the one with the highest average value at its leav nodes.

### An aside on subtrees

When making a naive search function which evaluated all subtrees, it came to my attention that there are many subtrees.

Given these definitions:
* **tree depth** the number of levels in the tree, counting the root node as 1, the children of the root node as level 2, and so on.
* **branch count** the number of children at each node before the maximum depth.
* **split** the number of children at each non-leaf node in the subtree.
* `f(tree depth, branch count, split) -> number of subtrees` a function which calculated the number of subtrees for given parameters.

you get the following numbers.

```
f(tree depth=3, branch count=2, split=1) -> 7
f(tree depth=3, branch count=2, split=2) -> 5
f(tree depth=3, branch count=3, split=1) -> 13
f(tree depth=3, branch count=3, split=2) -> 49
f(tree depth=3, branch count=3, split=3) -> 9
f(tree depth=3, branch count=4, split=1) -> 21
f(tree depth=3, branch count=4, split=2) -> 295
f(tree depth=3, branch count=4, split=3) -> 501
f(tree depth=3, branch count=4, split=4) -> 17
f(tree depth=3, branch count=5, split=1) -> 13
f(tree depth=3, branch count=5, split=2) -> 1211
f(tree depth=3, branch count=5, split=3) -> 13311
f(tree depth=3, branch count=5, split=4) -> 6481
f(tree depth=3, branch count=5, split=5) -> 33

f(tree depth=4, branch count=2, split=1) -> 15
f(tree depth=4, branch count=2, split=2) -> 26
f(tree depth=4, branch count=3, split=1) -> 40
f(tree depth=4, branch count=3, split=2) -> 7204
f(tree depth=4, branch count=3, split=3) -> 730
f(tree depth=4, branch count=4, split=1) -> 85
f(tree depth=4, branch count=4, split=2) -> 522151
f(tree depth=4, branch count=4, split=3) -> 503006005
f(tree depth=4, branch count=4, split=4) -> 83522
f(tree depth=4, branch count=5, split=1) -> 156
f(tree depth=4, branch count=5, split=2) -> 14665211
f(tree depth=4, branch count=5, split=3) -> many
```

## Redefining SDS to search for subtrees

Redefining SDS such that it would search for the optimal subtree was a bit conceptually tricky. I had to balance remaining faithful to the tree search variant, with remaining faithful to standard SDS, and also accurately representing the new problem.

The main issues were these:
1. There is no way to "randomly complete" a hypothesis in subtree search. Where a game tree always bottoms out, we are expecting to find solutions that do not reach the leaf nodes of the larger graph.
1. There is no "node indicated by an agent's hypothesis", each element of a hypothesis represents a **split** in a tree, indicating at least two children of a node, rather than just one in the game tree problem.
1. There is no "parent node" for any hypothesis as a hypothesis may represent a fairly complex subtree with many leaf nodes, there is not necessarily a single node which represents the "latest" or most recent split. A split that occurs high up in the subtree (closer to the root node) may have been added more recently than a split that occurs low in the subtree (further away from the root node).

This means that the form of the hypothesis must be considered very carefully, also the methods of "moving down" and "moving up" must be reconceived.

### Moving up and down become "splitting" and "pruning"

As an agent can't simply move up or down in their hypothesised (non-branching) path, I had to invent new actions, I decided to use splitting and pruning.

#### Splitting

Splitting means selecting a leaf node in the currently hypothesised subtree and adding two randomly selected children of the leaf node to the hypothesis. In cases where you are not specifically required to find a binary subtree then the number of children selected may be arbitrary.

This requires that hypotheses are represented in a way that the operation of identifying leaf nodes and replaching them with a collection of one or more of their children was feasible.

#### Pruning

Pruning means collapsing a previous split back into the parent node. To ensure that pruning had the same magitude of effect as splitting the only node which could be pruned were those whose children did not include any splits. Furthermore you could not prune a node which would lead to an invalid split. 

To avoid this problem pruning is only valid at what I am calling 'leaf splits'.

> If a simple hypothesis simply included a split at the root node 'r' to two child nodes 'ra' and 'rb' it would be legal to prune at 'r', removing 'ra' and 'rb' but you could not prune just one of either 'ra' or 'rb' as that would lead to a single child of 'r' which is no longer a binary tree.
> Similarly. If a hypothesis included a split at the root node 'r' to two child nodes 'ra' and 'rb' and a split at 'ra' to 'raa' and 'rab' you could not prune 'r' as it would be removing two splits (the split at 'r' and the split at 'ra'). You could only prune 'ra'. 

This requires that hypotheses are represented in a way that the operation of identifying the leaf splits and replacing a leaf split with the parent node was feasible.

#### Hypothesis representation

I settled on a hypothesis being an list of splits. Where the empty list represented no splitting and effectively indicated the root node. Hypotheses could therefore look like this.

```python
root_node: None

initial_split: [ra, rb]

evenly_balanced_splits: [[raa, rab], [rba, rbb]]

unevenly_balanced_splits: [[raa, [[rabaa, rabab], rabb]], [rba, rbb]]
```

#### SDS algorithm for subtree search

I struggled with many possible variants of testing and diffusion, what I ended up deciding to work with works as follows:
* **Initialisation** All agents hypothesise the empty list, which represents a minimal subtree and is equivalent of selecting just the root node.
* **Initial test** All agents become active with a probability P equal to the value at the root node.
* **Diffusion** Each inactive agent polls another agent.
  * **Polled is inactive** The polling agent prunes a split from their hypothesis
  * **Polled is active** The polling agent assumes the active agent's hypothesis plus a random split.

I have not implemented this variant yet, I ran out of time. But I have a method of conveneintly representing, creating, testing, and modifying hypotheses. So when I get down to implementing the actual SDS it shouldn't be so troublesome. This is, however, if and only if I have anticipated all problems.

This is rarely the case.

## Objections anticipated

1. This doesn't explicitely use multiple populations, I expect this will irritate The Bish. I am less concerned about this as I see separate populations as equivalent to restricted communication you see in lattice SDS. The only theoretical effect is has is to modify the probability distribution of which agent will be polled when randomly polling another agent. I also think even in cases where distinct populations are used you can often get nice effects by having the populations communicate in a limited way. Even Thomas's method allows for agents to move into different populations so it's quite a dynamic system anyhow.

1. This method of starting at the root node and recursively splitting means that certain hypotheses have a zero probability of being generated as it is required that all the intermediate hypotheses are strong enough to maintain stable clusters. Imagine for example that all the optimal solution existed at a certain level but that all the splits at the previous were so poor that agents never became active. This may turn out to be either an unrealistic situation (we expect more splits to represent situations that are more likely to overfit the data and hence score highly), and it may also be a good feature, maybe we don't want to search past such poor valued hypotheses.

1. This method might not work. My algorithm is completely untested and is far enough away from Standard SDS, Thomas's Tree Search and any published variant for its behaviour to be predicatable. I ended up at this variant because it was the first one that made any sense, I felt quite "funneled" down to this variant as anything else I thought of had some critical problem.

Feedback welcomed.
