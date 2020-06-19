
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

These experiments use the [SDS python library](https://github.com/AndrewOwenMartin/sds), which includes some standard methods of Diffusion and Testing, as well as Iteration and Halting, we will need to define some unique behaviours for the clustering and we will reuse some standard methods provided by the library.

The SDS library defines Standard SDS as follows:

``` python
def SDS(I, H):
    while not H():
        I()
```

Where `H` is a mode of halting, and `I` is a mode of iteration.

For now we will also use a simple mode of halting [fixed iterations halting](https://github.com/AndrewOwenMartin/sds/blob/46ec870a39b2a6b43938e21d3933c487e0b40c26/sds/standard.py#L180-L201) which halts after a given number of iterations.

We will use the standard method of iteration [synchronous iteration](https://github.com/AndrewOwenMartin/sds/blob/46ec870a39b2a6b43938e21d3933c487e0b40c26/sds/standard.py#L128-L139), which performs diffusion for all agents followed by testing for all agents.

``` python
def I_sync(D, T, swarm):

    def I():

        for agent in swarm:
            D(agent)

        for agent in swarm:
            T(agent)

    return I
```

As can be seen from the definition of synchronous iteration, it requires that two functions are passed `D` to define diffusion and `T` to define testing.

We will start with the standard method of diffusion [passive diffusion](https://github.com/AndrewOwenMartin/sds/blob/46ec870a39b2a6b43938e21d3933c487e0b40c26/sds/standard.py#L142-L157) in which active agents remain unchanged and inactive agents either select new hypotheses or copy the hypothesis of an active agent. Passive diffusion requires that the function `DH` is passed which is used for selecting new hypotheses, this is specific to the task of SDS Clustering so we will define it in the next section.

We will also start with the standard method of testing [boolean testing](https://github.com/AndrewOwenMartin/sds/blob/46ec870a39b2a6b43938e21d3933c487e0b40c26/sds/standard.py#L204-L213), so called as each test requires evaluating a boolean function. As with passive diffusion, boolean testing requires the definition of a task-specific function. This function `TM` is used for randomly selecting a 'microtest' function which performs a partial evaluation of the hypothesis.

## SDS Clustering #1: First attempt

The task-specific functions for SDS clustering are `DH` (for **H**ypothesis selection during the **D**iffusion phase) and `TM` (for **M**icrotest selection during the **T**est phase).

### DH: Random hypothesis selection

As the function for selecting a random hypothesis is determined by the dataset itself I make a function [`sds_ml.clustering.clustering.make_DH`](clustering.py#L39-L74) which returns a function `DH`. `make_DH` expects to be passed some information about the search space and returns `DH`. `DH` takes no arguments and returns a random hypothesis.

### TM: Random microtest selection

The function for selecting a random microtest is also tightly coupled to the dataset so I make a function [`sds_ml.clustering.clustering.make_boolean_TM`](clustering.py#L102-L132) which returns a function `TM`. `make_boolean_TM` gets passed the entire dataset and returns `TM`. `TM` takes no arguments and returns a random microtest.

The [`microtest`](clustering.py#L77-L99) function simply tests whether a single centroid is within the threshold distance of a given point.

### Putting it together

With the ability to define `TM` and `DH` and search spaces (with `make_a_problem_space`) I was ready to construct and SDS.

The full function can be seen in [clustering.example_basic](clustering.py#L140-L205), but the main composing of functions looks like this.

``` python
swarm = sds.Swarm(agent_count=agent_count)

H = sds.H_fixed(max_iterations)

TM = make_boolean_TM(
    points=points,
    dimension_count=dimensions,
    distance_metric=euclid_squared,
    threshold=threshold,
    rng=rng,
)

T = sds.T_boolean(TM)

DH = make_DH(points=points, dimension_count=dimensions, max_k=max_k, rng=rng)

D = sds.D_passive(DH, swarm, rng)

I = sds.I_sync(D, T, swarm)

sds.SDS(I=I, H=H)
```

You can run this function yourself with `python -m sds_ml.clustering.clustering basic` from the repository root directory.

Example output

```
2020-06-19 20:00:22 INFO __main__ Point cluster distribution: [(0, 5), (1, 47), (2, 36), (3, 12)]
2020-06-19 20:00:22 INFO __main__ Created problem with these centroids:
centroid #0: [0.742, 0.212, 0.567]
centroid #1: [0.739, 0.438, 0.581]
centroid #2: [0.182, 0.526, 0.782]
centroid #3: [0.069, 0.506, 0.084]
2020-06-19 20:00:22 INFO __main__ Running SDS for for 1000 iterations with 1000 agents.
2020-06-19 20:00:29 INFO __main__ Hypothesis #1 has 6 centroids.
centroid #0: [0.195, 0.565, 0.083]
centroid #1: [0.132, 0.425, 0.846]
centroid #2: [0.683, 0.190, 0.387]
centroid #3: [0.117, 0.538, 0.099]
centroid #4: [0.791, 0.436, 0.479]
centroid #5: [0.616, 0.493, 0.278]: 110
Hypothesis #2 has 7 centroids.
centroid #0: [0.034, 0.590, -0.054]
centroid #1: [0.666, 0.476, 0.731]
centroid #2: [0.070, 0.353, 0.189]
centroid #3: [0.791, 0.221, 0.538]
centroid #4: [0.348, 0.471, 0.832]
centroid #5: [0.138, 0.520, -0.007]
centroid #6: [0.309, 0.327, 0.333]: 92
Hypothesis #3 has 7 centroids.
centroid #0: [0.206, 0.490, 0.248]
centroid #1: [0.131, 0.590, 0.310]
centroid #2: [0.260, 0.492, 0.852]
centroid #3: [0.618, 0.371, 0.854]
centroid #4: [0.718, 0.410, 0.672]
centroid #5: [0.078, 0.323, 0.136]
centroid #6: [0.777, 0.187, 0.136]: 85
Hypothesis #4 has 7 centroids.
centroid #0: [0.429, 0.219, -0.056]
centroid #1: [0.196, 0.276, 0.392]
centroid #2: [0.321, 0.518, 0.022]
centroid #3: [0.638, 0.255, 0.645]
centroid #4: [0.733, 0.532, 0.657]
centroid #5: [0.632, 0.423, 0.200]
centroid #6: [0.049, 0.548, 0.843]: 85
Hypothesis #5 has 5 centroids.
centroid #0: [0.477, 0.202, 0.611]
centroid #1: [0.270, 0.486, 0.800]
centroid #2: [0.228, 0.364, 0.132]
centroid #3: [0.715, 0.384, 0.645]
centroid #4: [0.104, 0.334, 0.680]: 82
```

# Stuff to do.

I would like to compare the performance against pure random search. It would be interesting to compare the performance of anything against this most zen algorithm.

