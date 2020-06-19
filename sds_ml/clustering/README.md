
# SDS Clustering

This is a record of my progress with using SDS to cluster a potentially infinite, high-dimensionality dataset.

## Introduction

SDS takes the general form of a looping algorithm alternating between the Test phase and the Diffusion phase. In the Diffusion phase agents hypothesise solutions either by generating them at random or by copying them from active agents. In the Test phase agents determine whether or not they are active by performing a partial evaluation of their hypothesis. In short, think of the Diffusion phase as an opportunity for agents to select a hypothesis, and the Test phase as an opportunity for agents to decide whether they want to discard or share their current hypothesis.

Clustering is the problem of partitioning a set of data points so that similar points are in the same class. One method of doing this is called k-means clustering in which a number of "centroids" are chosen which best indicate the center of each putative cluster. In this algorithm the number of centroids to locate is given as a parameter.

The method of clustering with SDS borrows the notion of locating a number of centroids from k-means clustering. Similarly the number of centroids to fit is determined by a parameter but rather than being an integer, it is a range of integers.

## Procedure for SDS clustering

These parameters are passed to SDS clustering. 
* **k-range** (*K*) An interval representing valid numbers for *k*, the number of centroids to select.
* **data ranges** (*D*) The minimum and maximum values for each dimension of the data.
* **Agent count** (*A*) The number of agents in the swarm.
* **Halting criteria** A boolean function which will cause the algorithm to halt once it returns *true*.
* **Distance metric** A way of measuring the distance between a centroid and a data point, this can be assumed to be euclidean distance if not specified.
* **Accuracy threshold** The maximum acceptable distance between a centroid and a data point 

The algorithm then alternates between **Diffusion** and **Testing** until the **Halting criteria** are met. Examples of common **Halting criteria** are to halt after a fixed number of iterations, after a fixed amount of run time, or once a certain proportion of the swarm are active.

* **Diffusion** (in the first iteration all agents are *inactive*) Any *inactive* agent *a* polls another random agent *p*.
  * If *p* is *inactive*: *a* selects a new hypothesis at random by randomly generating an integer *k* in the interval *K*. The agent then generates *k* random points in uniformly at random within the data ranges *D*.
  * If *p* is *active*: *a* copies the hypothesis of *p*
* **Testing** Each agent selects a data point at random from the data set and finds the shortest distance between that point and any of their hypothesised centroids, if the distance is within the *accuracy threshold* they become *active*, else *inactive*.

This process will lead to the formation of clusters, where a cluster is a number of agents sharing the same hypothesis and the size of a cluster is the count of agents that share the hypothesis.
The score of a hypothesis is the mean average probability of an agent remaining active after **Testing**.
A cluster will not be stable if the score is less than 0.5, in which case is not likely to exist for more than a small number of iterations, if any.
The largest cluster will form at the hypothesis which has the greatest score.

## Housekeeping

Before I can run some examples I had to make some problem spaces, to do that, I used this algorithm.

[`sds_ml.clustering.problem.make_a_problem_space`](problem.py#L10-L63)

# Stuff to do.

I would like to compare the performance against pure random search. It would be interesting to compare the performance of anything against this most zen algorithm.

