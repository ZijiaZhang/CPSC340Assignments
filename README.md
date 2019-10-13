# CPSC340 Reviews
## Table of Contents
- [Basics](#basics)
  * [Steps for Data Mining](#steps-for-data-mining)
  * [Features](#features)
    + [Type of features](#type-of-features)
      - [Categorical Features](#categorical-features)
      - [Numerical Features](#numerical-features)
    + [Convert Features](#convert-features)
    + [Feature Aggregation](#feature-aggregation)
    + [Feature Selection](#feature-selection)
    + [Feature Transformation](#feature-transformation)
  * [Analize Data](#analize-data)
    + [Categorical Summary Statistics](#categorical-summary-statistics)
    + [Outliers](#outliers)
    + [Entropy as Measure of Randomness](#entropy-as-measure-of-randomness)
    + [Distance and Similarity](#distance-and-similarity)
    + [Limitations](#limitations)
  * [Visualization](#visualization)
    + [Basic Plots](#basic-plots)
  * [Supervised Learning](#supervised-learning)
    + [Naive Method: Predict Mode](#naive-method--predict-mode)

## Basics

### Steps for Data Mining
    
1) Identify data mining task. 
3) Collect data.
4) Clean and preprocess the data. 
5) Transform data or select useful subsets. 
6) Choose data mining algorithm. 
7) Data mining! 
8) Evaluate, visualize, and interpret results. 
9) Use results for profit or other goals

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

### Supervised Learning
Take features of examples and corresponding labels as inputs.
And Find a model that can accurately predict the labels of new examples.
- Input for an example is a set of features. 
- Output is a desired class label.

#### Naive Method: Predict Mode
Always Predict the mode.
