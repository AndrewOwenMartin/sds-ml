# Can SDS do linear classification?

## Experiment 1 - Single feature threshold categorisation

I started off simple, can SDS be used to find the single most predictive feature in a dataset.
I.e can it find the best parameters for `dimension`, and `threshold` in the function `microtest` shown below, such that it best predicts the class of an input row.

    def microtest(row, dimension, threshold):

        in_class = row[dimension] > threshold

        return in_class

To do this I used a Standard SDS:

    SDS(I=I_sync(D=D_passive(DH), T=T_boolean(TM)), H=H_fixed(2000))

Only `TM` and `DH` needed to be specified for this task.

### New hypothesis selection - DH

The hypothesis selection method returns a random `DimensionThreshold`.

    DimensionThreshold = collections.namedtuple(
        "DimensionThreshold",
        ("dimension", "operator", "threshold")
    )

Where `dimension` is the index of the chosen feature, `operator` is in `{operator.lt, operator.gt}` (less than or greater than) and threshold is one of the values in the dataset from the chosen dimension.

The variant is called `DH_plane` as it returns a region of a hyperspace defined as everything to one side of a plane.

    def DH_plane(dataset, rng):
        """ 
        Takes a "choice"able dataset where each row is expected to be of the same
        length.
        
        Each row must also be indexable and each value in a row is ordered with
        respect to all other values in the same column.

        Returns a single (feature_index, threshold, operator) tuple
        """

        def DH():

            random_row = rng.choice(dataset)

            feature_num = rng.randrange(len(random_row)-1)

            threshold = random_row[feature_num]

            operator = random.choice(sds_ml.operators)

            return DimensionThreshold(feature_num, operator, threshold)

        return DH

### Microtest selection - TM

The microtest selection method returns a partial function, formed by a generic microtest function applied to a randomly selected row from the dataset.

The generic function `microtest` takes a hypothesis in the form of a `DimensionThreshold` and a row from the dataset, and returns whether the class of the row matches the predicted class due to its position relative to plane defined by the hypothesis.

This whole operation, in short, evaluates the boolean expression 

    row class ==  hyp.operator(row[hyp.dimension], hyp.threshold)

Here is the full definition of `TM_plane`.

    def TM_plane(dataset, rng):
        """
        Takes a (feature_index, threshold, operator) hyp and a row.

        Applies operator to row[feature_index] and compares result to row[-1]
        """

        def microtest(hyp, row):

            feature = row[hyp.dimension]

            comparison = hyp.operator(feature, hyp.threshold)

            return comparison == row[-1]

        def TM():

            row = rng.choice(dataset)

            return functools.partial(microtest, row=row)

        return TM, microtest

## Experiment

I found some toy datasets here: [10 Standard Datasets for Practicing Applied Machine Learning](https://machinelearningmastery.com/standard-machine-learning-datasets/)

Two datasets looked likely:

1. Wine Quality Dataset. 11 input variables (real numbers), 1 output variable (integer in range [0,10]). 4898 observations, not balanced equally over the output classes.
2. Pima Indians Diabetes Dataset. 8 input variables (real numbers and integers), 1 output variable (boolean). 768 observations, not balanced equally over the output classes.

My current task was regarding boolean classification, so I went with the [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names).

First, an aside. The "Indians" of the "Pima Indians Diabetes Dataset" are North Americans, so we're in a tricky space when it comes to using the name "Indian" and so I'll say Pima where necessary. Having said that, here's a relevant quote from the book "1941: The Americas before Columbus" by Charles C. Mann.

> I abhor the term Native American," [Russel] Means declared in 1998.
> Matching his actions to his words, Means had joined and become prominent in
> an indigenous-rights group called the American Indian Movement. "We were
> colonised as American Indians," he wrote, "we were colonized as American
> Indians, and we will gain our freedom as American Indians, and then we will
> call ourselves any damn thing we choose" (At the same time, the common
> British usage of "Red Indian" to distinguish American natives from "East
> Indians" is unwelcome.)

The dataset has these features:

Input features

- Number of times pregnant.
- Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- Diastolic blood pressure (mm Hg).
- Triceps skinfold thickness (mm).
- 2-Hour serum insulin (mu U/ml).
- Body mass index (weight in kg/(height in m)^2).
- Diabetes pedigree function.
- Age (years).

Output features:

- Signs of Diabetes (0 or 1).

Running this variant against the Pima Indians Diabetes Dataset with 1000 agents converges in somewhere around 3000-5000 iterations, which on my computer is around 25 seconds.

The largest cluster is at the hypothesis ("Plasma glucose concentration a 2 hours in an oral glucose tolerance test" > 143.00).

Here's some stats, for a dataset of size 768 with 268 "positive" instances.

<table>
	<tr><td>Accuracy</td><td>0.75</td></tr>
	<tr><td>Cluster size</td><td>0.3</td></tr>
	<tr><td>True positive</td><td>126</td></tr>
	<tr><td>True Negative</td><td>450</td></tr>
	<tr><td>False positive</td><td>50</td></tr>
	<tr><td>False Negative</td><td>142</td></tr>
	<tr><td>Precision</td><td>0.716</td></tr>
	<tr><td>Recall</td><td>0.470</td></tr>
</table>

Where accuracy with a proportion of correctly guessed rows, precision is the proportion of "in class" guesses being correct, and recall is the proportion of all "in class" rows in the database that were correctly identified.

## Experiment 2 - Plane union categorisation

I allowed agents to choose up to one threshold per dimension, (e.g. A > x, B < y, C > z), but not (A > x, A < y, A > z) and take the OR of all comparisons, basically the union of a number of regions defined as being on one side of a plane.

I needed to redefine the hypothesis selection method DH and the microtest selection method TM.

### New hypothesis selection - DH

A hypothesis is now a set of `DimensionThreshold`s, each one for a different dimension. The number is chosen uniformly at random between 1 and the number of dimensions in the dataset.

    def DH_plane_union(dataset, rng):
        """
        Takes a "choice"able dataset where each row is expected to be of the same
        length.
        
        Each row must also be indexable and each value in a row is ordered with
        respect to all other values in the same column.

        Returns between one dimension threshold, or one dimension threshold per dimension, with no repeated dimensions.

        Returns an unordered set of (feature_index, threshold, operator) tuples
        """

        dimension_count = max(len(row)-1 for row in dataset)

        def DH():

            hyp_dim_count = rng.randint(2, dimension_count)

            hyp_dims = rng.sample(range(dimension_count), hyp_dim_count)

            hyp = frozenset(
                DimensionThreshold(
                    dimension=dimension,
                    operator=rng.choice(sds_ml.sds_ml.operators),
                    threshold=rng.choice(dataset)[dimension],
                )
                for dimension
                in sorted(hyp_dims)
            )

            return hyp

        return DH

### Microtest selection - TM

The microtest selects a random row from the dataset and returns *true* if any of the `DimensionThreshold` comparisons return *true*.

    def TM_plane_union(dataset, rng):

        def microtest(hyp, row):

            comparison = any(
                plane.operator(row[plane.dimension], plane.threshold)
                for plane
                in hyp
            )

            return comparison == row[-1]

        def TM():

            row = rng.choice(dataset)

            return functools.partial(microtest, row=row)

        return TM, microtest

Interestingly for the first few thousand iterations, you tend to see exactly the same result ("Plasma glucose concentration a 2 hours in an oral glucose tolerance test." > 143.00), even if you force the hypotheses to have at least two dimensions you see a combination of the same dimension plus a junk dimension, e.g. ("Plasma glucose concentration a 2 hours in an oral glucose tolerance test." > 143.00 OR "Diastolic blood pressure (mm Hg)." < 0).

After between 1000 and 3000 iterations a new cluster forms at ("Plasma glucose concentration a 2 hours in an oral glucose tolerance test" > 139.00 OR "Body mass index (weight in kg/(height in m)^2)." > 45.30)

Here are some stats for the dataset of size 768 with 268 "positive" instances.
<table>
	<tr><td>Accuracy</td><td>0.76</td></tr>
	<tr><td>Cluster size</td><td>0.58</td></tr>
	<tr><td>True positive</td><td>147</td></tr>
	<tr><td>True Negative</td><td>433</td></tr>
	<tr><td>False positive</td><td>67</td></tr>
	<tr><td>False Negative</td><td>121</td></tr>
	<tr><td>Precision</td><td>0.687</td></tr>
	<tr><td>Recall</td><td>0.549</td></tr>
</table>

# Experiment 3 - Plane intersection categorisation

I then tried "plane intersection", where agents would choose up to one threshold per dimension, and take the AND of all comparisons, basically the intersection of a number of regions defined as being one side of a plane.

I only needed to redefine TM

### Microtest selection - TM

The microtest selects a random row from the dataset and returns *true* if any of the `DimensionThreshold` comparisons return *true*.

    def TM_plane_intersection(dataset, rng):

        def microtest(hyp, row):

            comparison = all(
                plane.operator(row[plane.dimension], plane.threshold)
                for plane
                in hyp
            )

            return comparison == row[-1]

        def TM():

            row = rng.choice(dataset)

            return functools.partial(microtest, row=row)

        return TM, microtest

After 5000 iterations with 10000 agents the largest cluster is at ((insulin < 545.00) AND (bmi > 30.10) AND (glucose_concentration > 126.00)

Here are the stats for this variant.
<table>
	<tr><td>Accuracy</td><td>0.77</td></tr>
	<tr><td>Cluster size</td><td>0.32</td></tr>
	<tr><td>True positive</td><td>145</td></tr>
	<tr><td>True Negative</td><td>443</td></tr>
	<tr><td>False positive</td><td>57</td></tr>
	<tr><td>False Negative</td><td>123</td></tr>
	<tr><td>Precision</td><td>0.718</td></tr>
	<tr><td>Recall</td><td>0.541</td></tr>
</table>

My next interest is in defining hypotheses as the union of a set of intersections. This is probably when we'll reach the limit of uniformly random sampling, and population sampling will come in, but I'll try an example with uniformly random selection and see where I get.

Population sampling is where random numbers are sampled from a distribution representing the current distribution of active agents rather than sampling a uniform distribution. For example, rather than selecting a dimension at random, a dimension will be selected by randomly selecting an agent, and if the selected agent is active, using a dimension from their hypothesis, if the selected agent is inactive, a dimension is selected uniformly at random.

Please send questions or comments to [andrew@aomartin.co.uk](mailto:andrew@aomartin.co.uk?subject=sds-ml), but do so on the understanding that I may post the entire content here.
