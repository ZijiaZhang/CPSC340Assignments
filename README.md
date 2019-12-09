# CPSC340 Reviews
## Table of Contents
- [CPSC340 Reviews](#cpsc340-reviews)
  - [Table of Contents](#table-of-contents)
  - [Basics](#basics)
    - [Steps for Data Mining](#steps-for-data-mining)
    - [Features](#features)
      - [Type of features](#type-of-features)
        - [Categorical Features](#categorical-features)
        - [Numerical Features](#numerical-features)
      - [Convert Features](#convert-features)
      - [Feature Aggregation](#feature-aggregation)
      - [Feature Selection](#feature-selection)
      - [Feature Transformation](#feature-transformation)
    - [Analize Data](#analize-data)
      - [Categorical Summary Statistics](#categorical-summary-statistics)
      - [Outliers](#outliers)
      - [Entropy as Measure of Randomness](#entropy-as-measure-of-randomness)
      - [Distance and Similarity](#distance-and-similarity)
      - [Limitations](#limitations)
    - [Visualization](#visualization)
      - [Basic Plots](#basic-plots)
  - [Supervised Learning](#supervised-learning)
    - [Naive Method: Predict Mode](#naive-method-predict-mode)
    - [Decision Trees](#decision-trees)
      - [Decision Stump](#decision-stump)
      - [Measure of goodness: Accuracy score](#measure-of-goodness-accuracy-score)
      - [Greedy recursive splitting](#greedy-recursive-splitting)
    - [IID Assumptions.](#iid-assumptions)
    - [Training vs Test Error](#training-vs-test-error)
    - [Validation Error](#validation-error)
    - [Optomization bias](#optomization-bias)
    - [Cross Validation](#cross-validation)
    - [Probabilistic Classifiers](#probabilistic-classifiers)
      - [naive Bayes](#naive-bayes)
    - [KNN](#knn)
      - [K on fundamental trade-off](#k-on-fundamental-trade-off)
      - [Charasteristics](#charasteristics)
      - [Problems](#problems)
      - [Parametric vs. Non-Parametric Models](#parametric-vs-non-parametric-models)
    - [Norms](#norms)
    - [Encouraging Invariance](#encouraging-invariance)
    - [Ensemble Method](#ensemble-method)
      - [Types of Ensemble Method](#types-of-ensemble-method)
      - [Averaging](#averaging)
        - [Random Forests](#random-forests)
  - [Unsupervised Learning](#unsupervised-learning)
    - [Clustering](#clustering)
      - [K-means](#k-means)
        - [Cost of K-Means](#cost-of-k-means)
        - [Application of K-means](#application-of-k-means)
      - [Density Based Clustering](#density-based-clustering)
      - [Hierarchical Clustering](#hierarchical-clustering)
      - [Agglomerative (Bottom-Up) Clustering](#agglomerative-bottom-up-clustering)
      - [Biclustering](#biclustering)
    - [Outliers Detection](#outliers-detection)
      - [Model-based methods.](#model-based-methods)
      - [Graphical approaches.](#graphical-approaches)
      - [Cluster-based method](#cluster-based-method)
      - [Distance-based methods](#distance-based-methods)
      - [Supervised-learning methods](#supervised-learning-methods)
    - [Supervised Learning (Regressions)](#supervised-learning-regressions)
      - [Linear regression based on squared error.](#linear-regression-based-on-squared-error)
      - [Non-linear Regression](#non-linear-regression)
      - [Gradiant Decent](#gradiant-decent)
    - [Robust Regression](#robust-regression)
    - [Complexity Penalties](#complexity-penalties)



## Basics

### Steps for Data Mining
    
1) Identify data mining task. 
2) Collect data.
3) Clean and preprocess the data. 
4) Transform data or select useful subsets. 
5) Choose data mining algorithm. 
6) Data mining! 
7) Evaluate, visualize, and interpret results. 
8) Use results for profit or other goals

### Features
#### Type of features
##### Categorical Features
The features that is not a number, like job and city. 
##### Numerical Features
The Feature that is a number
#### Convert Features
- Categorized
		Make category as a number.
- Text
		Bag of Words
- Graph
		Grayscale intensity
		Adjacency Matrix
#### Feature Aggregation
Combine features to form new features.
e.g. City to Provinces
#### Feature Selection
Remove irrelevent feature
#### Feature Transformation
- Discretization
		Make continuous data to categories.
- Square, exponentiation, logarithm...
- Scaling

### Analize Data
#### Categorical Summary Statistics
- Mean: average value. 
- Median: value such that half points are larger/smaller. 
- Quantiles: value such that ‘k’ fraction of points are larger
- Range: minimum and maximum values. 
- Variance: measures how far values are from mean.  (Square root of variance is “standard deviation”)
- Intequantile ranges: difference between quantiles.
#### Outliers
Mean and std is more sensitive to ourliers.

#### Entropy as Measure of Randomness
- Low entropy means “very predictable”. 
- High entropy means “very random”
- For $k$ values Minimum value is 0, maximum value is $\log(k)$.
- For continuous dataset, normal distribution has highest entropy
- For categorical dataset: uniform distribution has highest entropy.

#### Distance and Similarity
- Hamming distance: Number of elements in the vectors that aren’t equal. 
- Euclidean distance:  How far apart are the vectors? 
-  Correlation:  Does one increase/decrease linearly as the other increases?  Between -1 and 1.

#### Limitations
Summary statistic can be misleading. Dataset that are very different can have same summary statistic.

### Visualization

#### Basic Plots
- Histogram
- Box Plot
- Matrix Plot 
May be able to see trends in features.
- Scatterplot

## Supervised Learning
Take features of examples and corresponding labels as inputs.
And Find a model that can accurately predict the labels of new examples.
- Input for an example is a set of features. 
- Output is a desired class label.

### Naive Method: Predict Mode
Always Predict the mode.
### Decision Trees
#### Decision Stump
A simple decision tree with 1 spliting rules:

If feature $x >x_0$ 
	predict $y_0$ 
otherwise 
	predict $y_1$
	
	
This will take $O(ndk)$ with an instance of $n$ examples, $d$ features and $k$ thresholds.
$O(nd)$ if all features are binary.
$O(n^2d)$ if all our features have unique values. 
#### Measure of goodness: Accuracy score
Score = $\frac{\text{Correct Inputs}}{\text{Total number of Examples}}$

#### Greedy recursive splitting
 1. Find the decision stump with the best score (information gain).
 2. Find the decision stump with the best score on the two new datasets.
 3. Stop if all leaves have the  same label, or reaches the maximum depth.
 
### IID Assumptions.
Usually we would Assume that the training set and test set follows:

- All examples come from the same distribution
- The example are sampled independently

### Training vs Test Error
Test Error = Approximation Error + Training Error

- **Simple model** might have a high training Error but have a low approximation error.
- **Complex model** might have a low trainging Error but have a high approximation error.

e.g. The deeper the decision tree is the more complex it is. If we have a decision tree with depth $\infty$, every thing in training set will be classified correctly. But it might perform poorly in Test Set.

### Validation Error
We can use part of training data to approximate the test error, helping us pick the hyper parameter.
But if we look use the validation set too much, we might introduce optimization bias.

### Optomization bias
We might tried too many models on the validation set too much and by chance one of them accidently have a low validation error. So we **overfit** the validation set.

e.g. Consider a multiple choice test with 10 questions.
Fill a exam randomly, expect grade => 25%
Fill 2 exams randomly, expect max grade => 33%
Fill 10000 exams randomly, expect max grade => 82%

- **Optimization bias is small if you only compare a few models.**
- **Optimization bias shrinks as you grow size of validation set.**

### Cross Validation
Split the trainging data to k parts. Train the model on k-1 parts, and compute the validation error on the other part.
Take the average of the k errors to approxmate the test error.

**As k get larger, the result gets more accurate but more expensive**

### Probabilistic Classifiers
#### naive Bayes
- ***Identify spam emails***
Here, $X_i$ is features of the email (bag of words).

  v1 = $p(y_i =\text{spam}|x_i) = \frac{p(x_i|y_i=\text{spam})p(p_i = \text{spam})}{p(x_i)}$
  
  v2 = $p(y_i = \text{not spam}|x_i) = \frac{p(x_i|y_i=\text{not spam})p(p_i = \text{not spam})}{p(x_i)}$

We would compare the value of v1 and v2, if v1 > v2 we say it is spam. Otherwise, not spam.
But $p(x_i|y_i=\text{spam})$ is hard to compute, so we assume that each word is independent. That is <br>
$p(x_i|y_i=\text{spam}) = \prod_{j=0}^d p(x_i^j|y)$


- Laplace Smoothing
Fix the problem that if you have no training example with that feature value, will result in a 0 probablility.

  $\frac{\text{Spam message with w} + 1}{\text{Spam messages} +2}$

### KNN
For a new example x, predict it with the same value as the training example that is nearest to it.

- Model gets more complicated as ‘k’ decreases.
- Model gets more complicated as ‘n’ increases.

#### K on fundamental trade-off
As ‘k’ grows, training error increase and approximation error decreases.

#### Charasteristics
- No training phase
- Predictions are expensive: O(nd) for one example.
- Storage is expensive: Store all training data.

#### Problems
- Features have very different scales
- Need exponentially more points to ‘fill’ a high-dimensional volume.
#### Parametric vs. Non-Parametric Models
- **Parametric**
Have fixed number of parameters: trained “model” size is O(1) in terms ‘n’.
e.g. Decision Tree
- **Non-Parametric**
Number of parameters grows with ‘n’: size of “model” depends on ‘n’

e.g. KNN

### Norms
- L1 Norm
	$\| r\|_1 = \sum_{j=1}^d|r_j|$
- L2 Norm
$\| r\|_2 = \sqrt{\sum_{j=1}^d|r_j^2|}$
- L$\infty$ Norm
$\| r\|_\infty = \max\{|r_j|\}$

### Encouraging Invariance
Add transformed data during training to avoid small translation during tests.

### Ensemble Method
Classifiers that have classifiers as input.
- Often have higher accuracy than input classifiers.
#### Types of Ensemble Method
- Boosting (improve trainging error)
- Averaging (improves approximation error)

#### Averaging
- Input to averaging is the predictions of a set of models
- If the models make independent errors. The chanses that the averaing is wrong is lower than other models.
1. A simple model of averaging: Take mode of predictions.
![alternative text](./images/EnsembleMethod.PNG)
2. Stacking
![alternative text](./images/Stacking.PNG)

##### Random Forests
Average a set of deep decision trees. 

- Generate Trees with independent Errors
  - Bootstrap Sampling <br> Generate several bootstrap samples an d fit classifier to each boot strap sample. Average the prodictions.
  - Random Trees <br> For each split in a random tree model, randomly sample a small number of possible features, and condider these random features for spliting.


## Unsupervised Learning
### Clustering
- Input: set of examples described by features $x_i$
- Output: an assignment of examples to ‘groups’.
- Goal: 
  - Examples in the same group should be ‘similar’.
  - Examples in different groups should be ‘different’.


#### K-means
**Input:**
  
  The number of clusters ‘k’ (hyper-parameter).
  
  Initial guess of the center (the “mean”) of each cluster.

```
1. Assign each data point to closest mean.
2. Update the means bases on the assignment.
3. Repead untill convergence.

When get a new test example, assign it to the nearest mean.
```

**Note:**

K-Means is Garenteed to converge.

**Issues:**

Each example is assigned to one (and only one) cluster.

It may converge to sub-optimal solution. 

Clusters have to be convex.

**Solusion:**

Try several different random start points and choose the best.

##### Cost of K-Means
$O(ndk)$

##### Application of K-means
Represent the data with the cluster means.

#### Density Based Clustering
Clusters are defined by “dense” regions. Examples in non-dense regions don’t get clustered.

**Inputs:**

Epsilon($\epsilon$): Distance we use to decide if another point is a “neighbour”.

MinNeighbours: number of neighbours needed to say a region is “dense”

A core point is defined by: The number of neibours $\geq$ minNeighbours
```
For each example xi:
  If xi is already assigned to a cluster:
    do nothing.
  else Test whether xi is a ‘core’ point
    If xi is not core point, 
      do nothing (this could be an outlier).
    If xi is a core point
      make a new cluster and call the “expand cluster” function.
```

```
“Expand cluster” function:
Assign to this cluster all xj within distance ‘ε’ of core point xi to this cluster.
For each new “core” point found, call “expand cluster” (recursively).
```
**Issues:**

Some points are not assigned to a cluster.

Ambiguity of boundary points.

Sensitive to the choice of ε and minNeighbours.

With new Data Point, finding cluster is expensive.

#### Hierarchical Clustering

Produces a tree of clusterings.

- Each node in the tree splits the data into 2 or more clusters.
- Much more information than using a fixed clustering.
- Often have individual data points as leaves.

#### Agglomerative (Bottom-Up) Clustering

```
1. Starts with each point in its own cluster.
2. Each step merges the two “closest” clusters.
3. Stop with one big cluster that has all points.
```

- Needs a “distance” between two clusters.
- Cost is $O(n^3d)$
- Like Reversed Hierarchical Clustering

#### Biclustering
Cluster the training examples and features


### Outliers Detection

#### Model-based methods.
Fit a probabilistic model.

Outliers are examples with low probability.

e.g. The data with |Z score| > 4 is an outlier.

#### Graphical approaches.
Look at a plot of the data.

Human decides if data is an outlier.


- Box plot
  - Can only plot 1 variables at a time.
- Scatterplot
  - Only 2 variables at a time
- Scatterplot array
  - Look at all combinations of variables.
  - Still only 2 variables at a time.

#### Cluster-based method

Cluster the data

Find points that don’t belong to clusters.

- K-means
  - Find points that are far away from any mean
  - Find clusters with a small number of points
- Density-based clustering
  - Outliers are points not assigned to cluster
- Hierarchical clustering
  - Outliers take longer to join other groups

#### Distance-based methods

- KNN outlier detection
  - For each point, compute the average distance to its KNN
  - Choose points with biggest values (or values above a threshold) as outliers
- Local Distance-Based Outlier Detection
  - $\frac{\text{average distance of 'i' to its KNNs}}{\text{average distance of neighbours of i to their KNNs}}$
#### Supervised-learning methods

- We can find very complicated outlier patterns.
- We need to know what outliers look like.
- We may not detect new “types” of outliers

### Supervised Learning (Regressions)

#### Linear regression based on squared error.
Linear regression makes predictions using linear function of $x_i$.

$\hat{y_i} = w x_i$ Where w is the weight of $x_i$.

We measure the sum of suqared errors. 

$f(w) = \frac{1}{2}\sum_{i=1}^n(w^Tx_i-y_i)^2$

And we want to minimize this Error. We calculate the diravitives. 

$\nabla f(w) = [\sum_{i=1}^n(w^Tx_i-y_i)x_{i1}, \sum_{i=1}^n(w^Tx_i-y_i)x_{i2}, \sum_{i=1}^n(w^Tx_i-y_i)x_{i3}, \dots , \sum_{i=1}^n(w^Tx_i-y_i)x_{id}]^T$

We can have $W =(X^TX) \setminus (X^Ty)$ to have $\nabla f(w) = 0$

**Derive the diravitives**

$f(w) = \frac{1}{2} \|Xw-y\|^2 = \frac{1}{2} \sum_{i=1}{n}(w^Tx_i-y_i)^2 \\ = \frac{1}{2} w^TX^TXw-w^TX^Ty+\frac{1}{2}y^Ty$

$\nabla f(w) = X^TXw - X^Ty$

So, when $W =(X^TX) \setminus (X^Ty)$ we have $\nabla f(w) = 0$

**Note**

The solusion might not be unique.

e.g. 
| x1 | x2 | y |
| :---: |:---:|:--:|
| 1 | 1 | 3 |
| 2 | 2 | 6 |

So one solusion is $w_1 = 3$, $w_2 = 0$. And the other one is $w_1 = 0$, $w_2=3$.

#### Non-linear Regression
Now, we can have $\hat{y_i} = w_0 + w_1x_i + w_2x_i^2$ by doing changing the features.

$X = \begin{pmatrix} x_1 \cr x_2 \cr x_3 \end{pmatrix}$ 
$Z = \begin{pmatrix} 1 & x_1 & x_1^2 \cr 1 & x_2 & x_2^2  \cr 1 & x_3 & x_3^2  \end{pmatrix}$

The new vector is v, $\hat{y_i} = v^Tz_i$

- As the polynomial degree increases, the training error goes down
- But approximation error goes up: we start overfitting with large ‘p’.

#### Gradiant Decent

When d is large, solving the system will take $O(d^3)$, we can use the Gradiant Decent method.

```
– It starts with a “guess” w0
– It uses the gradient ∇ f(w0) to generate a better guess w1
– It uses the gradient ∇ f(w1) to generate a better guess w2
– It uses the gradient ∇ f(w2) to generate a better guess w3
…
– The limit of wt as ‘t’ goes to ∞ has ∇ f(wt) = 0.
```

$w_1 = w_0 - \alpha_0\nabla f(w_0)$

This decreases ‘f’ if the “step size” $\alpha_0$ is small enough.

Usually, we decrease $α_0$if it increases ‘f’ (see “findMin”).

The cost is $O(ndt)$ for t iterations.

**Limitations**

- If the function is not convex, we might converge to local min.

**Convex function**
- 1-variable, twice-differentiable function is convex iff f’’(w) ≥ 0 for all ‘w’.
- A convex function multiplied by non-negative constant is convex.
- Norms and squared norms are convex.
- The sum of convex functions is a convex function.
- The max of convex functions is a convex function.
- Composition of a convex function and a linear function is convex.


### Robust Regression
If we have a outlier, the linear regression will be sensitive to those.

Robust regression objectives focus less on large errors (outliers).
- For example, use absolute error instead of squared error:
$f(w) = \sum_{i=1}^n |w^Tx_i - y_1|$

$f(w) = \|Xw - y\|_1$

We have a smooth approximation to make it differentiable.

$f(w) = \sum_{i=1}^n h(w^Tx_i - y_i)$

$h(r_i) = \begin{cases} \frac{1}{2} r_i^2 &\text{for} |r_i|\leq \epsilon \\ \epsilon(|r_i| - \frac{1}{2} \epsilon) & \text{otherwise} \end{cases}$

Non-convex errors can be very robust:
- Not influenced by outlier groups.
- But non-convex, so finding
global minimum is hard.
- Absolute value is “most robust”
convex loss function

### Complexity Penalties

We might introduce Penalties to avoid overfitting.

e.g. $score(p) = \frac{1}{2} \|Z_pv - y\|^2 + \lambda k$ where k is degree of freedom.

The score will relative high if we are having a high k.

## Feature Selection
Select the feature that is most "important".
### Association
Compute correlation between feature values $x_j$ and ‘y’.
- Say that ‘j’ is relevant if correlation is above 0.9 or below -0.9.

Usually gives unsatisfactory results as it ignores variable interactions:
### Regression Weight
Fit regression weight 'w' based on all features. Take all features 'j' where $|w_j|$ is greater than a threshold.
**Issue:** 
1. If two variable are equal, any one might be relevent and the other is not.
$\hat{y_i} = w_1*d_1 + w_2*d_2 = 0*d_1 + (w_1+w_2)*d_2$
2. If there are 2 irrelevent feature, we might think they are both relevent:
	$\hat{y+i} = 0*d_1 + 0*d_2 = 10000*d_1 + -10000*d_2$

### Search and Score
1. Define score function f(S) that measures quality of a set of features ‘S’.
2. Now search for the variables ‘S’ with the best score.


Compute score with feature {{1},{2},{3}, {1,2}, {1,3}, {2,3}, {1,2,3}}

**Issue:** If we use validation error for the score, we might have an issue with optimization bias due to large number of sets ($2^d$).

To solve this issue, we will apply "Number of Features" penalties. 
$Score(S) = \frac 1 2 \sum_{i=1}^n(W_s^Tx_{is} - y_i)^2 + size(S)$ This makes that if the S have similar error, we would prefer the smaller set. Usually we would use L0-norm instead of function size.

## Forward Selection 
• In search and score, it’s also just hard to search for the best ‘S’. ($2^d$ possibilities)
(Like in CPSC304).

1. Start with an empty set of features, S = [ ]. 
2. For each possible feature ‘j’: 
- Compute scores of features in ‘S’ combined with feature ‘j’.
4. Find the ‘j’ that has the highest score when added to ‘S’. 
5. Check if {S ∪ j} improves on the best score found so far. 
6. Add ‘j’ to ‘S’ and go back to Step 2. 
	- A variation is to stop if no ‘j’ improves the score over just using ‘S’

## Regularization
### L2-Regularization
$f(w) = \frac 1 2 \|Xw - y\|^2 + \frac \lambda 2 \|w\|^2$
Objective balances getting low error vs. having small slopes $w_j$

- Regularization increases training error.
- Regularization decreases approximation error.

It should be in the same unit in regularization so they will have same effect.

### Standardizing Features
– For each feature: 
1. Compute mean and standard deviation: 
2. Subtract mean and divide by standard deviation (“z-score”)

In predication: use mean and standard deviation of training data.

## Non-paramatric bases

### Gaussian RBF
For each data point, we have a corresponding normal destribuation. 
1. We use ‘n’ bumps (non-parametric basis). 
2. Each bump is centered on one training example x_i 
 3. Fitting regression weights ‘w’ gives us the heights (and signs). 
 4.  The width is a hyper-parameter (narrow bumps == complicated model).

**Regularization**
We calculate Z, so we can use L2-regularization on W. and we can use cross validation to choose $\sigma$ (width of noramal distribution) and $\lambda$ (regularizer)

Hyper-Parameter Optimization 
- In this setting we have 2 hyper-parameters (𝜎 and λ). 
- More complicated models have even more hyper-parameters. 
	- This makes searching all values expensive (increases over-fitting risk). 
- Leads to the problem of hyper-parameter optimization. 
	-  Try to efficiently find “best” hyper-parameters. 
-  Simplest approaches: 
	- Exhaustive search: try all combinations among a fixed set of σ and λ values. 
	-  Random search: try random values


### L1-Regularization
Like L2-norm, it’s convex and improves our test error. 
 Like L0-norm, it encourages elements of ‘w’ to be exactly zero.
 L1-Regularization sets values to exactly 0

L2-Regularization: 
- Insensitive to changes in data. 
- Decreased variance: 
	-  Lower test error. 
- Closed-form solution. 
-  Solution is unique. 
-  All ‘wj ’ tend to be non-zero. 
-  Can learn with linear number of irrelevant features. 
	-  E.g., only O(d) relevant features. 
L1-Regularization: 
- Insensitive to changes in data.
-  Decreased variance: 
	-  Lower test error. 
- Requires iterative solver. 
-  Solution is not unique. 
-  Many ‘wj ’ tend to be zero. 
-  Can learn with exponential number of irrelevant features. 
	-  E.g., only O(log(d)) relevant features. Paper on this result by Andrew Ng

### Ensemble Feature Selection
We can use esemble method for feature selection. It is usually used to reduce false positives or false negatives.

## Linear Classifier
We can design a classifier that is $y_i = w1x_{i1} + w2x_{i2} + \dots + w_dx_{id}$. So that $y_i = 1$ for one class and $y_i = -1$ for the other class.

**Issue**: the issue is that the least square might peanalize the classifier being too right. One way to solve this is using ***0,1 loss*** which is how many samples is classified correctly.
### Perceptron Algorithm

- Start with $w_0 = 0.$ 
- Go through examples in any order until you make a mistake predicting yi . 
	-  Set $w_{t+1} = w_t + y_i x_i$. 
-  Keep going through examples until you make no errors on training data.

--If a perfect classifier exists, this algorithm finds one in finite number of steps

### 0-1 Loss
- Even if the 0-1 Loss can solve the issue, it is not a convex function.
	- It’s easy to minimize if a perfect classifier exists (perceptron). 
	- Otherwise, finding the ‘w’ minimizing 0-1 loss is a hard problem
- Gradient is zero every where: so no "direction".

### Convex approximation 0-1 Loss
With 0-1 Loss, 
1. w = 0 always gives you minimum possible f.
2. non-convex.
#### Hinge Loss
- $max\{0, 1- y_iw^Tx_i\}$
- ** Support vector machine** (SVM) is hinge loss with L2-regularization

#### Logistic Loss
- Approximate $max\{0, -y_iw^tx_i\}$ with $\log(exp(0) + exp (-y_iw^Tx_i)) = \log(1 + exp(-y_iw^Tx_i))$
- With w = 0, gives an error of log(2) instead of 0.
- Convex and differentiable: minimize this with gradient descent.

These two methods has:
1. Fast training and testing.
2. $w_j$ are easy to understand
3. Often get a good test error.
4. Smoother predictions than random forests.


### Predictions vs. Probabilities
We used sign function to make predictions. So it maps $w^Tx_i$ to +1 and -1. 

But for probabilities, we want to map to range [0,1].

We use sigmoid function:

$h(z_i) = \frac 1 {1+exp(-z_i)}$
We can use this as the probability that the sample belongs to one class. (“probability that e-mail is important”)

### Multi-Class Linear Classification
#### One vs all
– “One vs all” is a way to turn a binary classifier into a multi-class method

We have one classifier for each kind of class. And we can take the one with the highest score for the result of the prediction.

X = [n*d] Matrix 
W = [k * d] Matrix
- In W, for each row, is a linear classifier for one class. 
- Would pick maximum 'c' for $w_c^Tx_i$. 

**Issues**
Only works in convex reginons, 
Scores are not conpareable as the classifier are not realted to each other. 

### Multi-Class SVM
To solve the above problem, we want for each training sample:
- $w_{y_i}^Tx_i > w_c^Tx_i$ for all c that is not the correct label $y_i$

To enforce this, we use the methos similar to SVM that is $w_{y_i}^Tx_i\geq w_c^Tx_i + 1$ for all $c\neq y_i$.

So there are 2 pisible losses:
1. $\sum_{c\neq y_i} max\{0,1-w_{y_i}x_i+w_c^Tx_i\}$
2. $\max_{c\neq y_i} \{max\{0,1-w_{y_i}x_i+w_c^Tx_i\}\}$

- The "Sum: rule penalizes for each ‘c’ that violates the constraint.
- “Max” rule penalizes for one ‘c’ that violates the constraint the most

If we add L2-regularization, both are called multi-class SVMs

### Multi-Class Logistic Regression

$w_{y_i}^Tx_i \geq \max_c\{w_c^Tx_i\}$ 
which is equivalent to : $0 \geq -w_{y_i}^T x_i + \max_c\{w_c^Tx_i\}$

- We want the right side to be as small as possible.

Use log-sum-exp  on $\max_c$, we have:

$-w_{y_i}^Tx_i + \log(\sum_{c=1}^k \exp(w_c^Tx_i))$

We have the **softmax loss**.

$f(W) = \sum_{i=1}^n[-w_{y_i}^Tx_i + \log(\sum_{c=1}^k \exp(w_c^Tx_i))] + \frac \lambda 2 \sum_{c=1}^k \sum_{j=1}^d w_{c_j}^2$

- When k=2, equivalent to using binary logistic loss.

### Prediction
We have W = [k*d] and X = [n*d]

So predictions are maximum column indices of $XW^T$ (which is ‘n’ by ‘k’).


## Feature Engineering
The best features may be dependent on the model you use.

- Counting-based methods
	- Discretization
- Distance-based methods
	- Standardization
- Regression-based methods
	- Non-Linear Transformations

### Features for a santence.
1. String of chars
	Lose no information
2. Bag of words
	Count how many times each word appears.
3. n-grams (bigram, trigram)
	Ordered set of two words, gives more context.
### Global and local Features

**“Personalized” Important E-mails**
The global feature will change the predication for every user, but the local features will only change the prediction to specific users.

|340(Any User)|340(user?)|
| :---:|:---: |
|1| User1|
|1|User1|
|1|User2|
|0|--|
|1|User3|

We can apply feature transformation to it: 
|340(Any User)|340(user1)|340(user2)|
| :---:|:---: |:--:|
|1| 1|0|
|1|1|0|
|1|0|1|
|0|0|0|
|1|0|0|
