# Chapter 2

**First Step**: Frame The Problem.

- The first question to ask your boss is what exactly the business objective is.

- How does the company expect to use and benefit from this model?

>  Knowing the objective is important because it will determine
> how you frame the problem, which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it.

Your boss answers that your model’s output will be fed to another machine learning system along with many other signals.

![](/home/moaaz/snap/marktext/9/.config/marktext/images/2024-06-25-08-24-08-image.png)

> This downstream system will determine whether it is worth
> investing in a given area. Getting this right is critical, as it directly affects revenue.

- The next question to ask your boss is what the current solution looks like.

our boss answers that the district housing prices are currently estimated manually by experts This is costly and time-consuming, and their estimates are not great.

<p align="center"> <b>Pipelines</b>
</p>

A sequence of data processing components is called a data pipeline. Pipelines are very
common in machine learning systems, since there is a lot of data to manipulate and
many data transformations to apply.

---------

![](/home/moaaz/snap/marktext/9/.config/marktext/images/2024-06-29-17-11-51-image.png)

![](/home/moaaz/snap/marktext/9/.config/marktext/images/2024-06-29-17-17-08-image.png)

![Euclidean Distance: Advantages & Limitations | BotPenguin](https://cdn.botpenguin.com/assets/website/Euclidean_Distance_1_59a98c213f.png)

![Difference between Manhattan distance and Euclidean distance - Basics of  control engineering, this and that](https://taketake2.com/ne1615_en.png)

| Feature                      | Euclidean Distance                                                                          | Manhattan Distance                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Definition**               | The straight-line distance between two points.                                              | The sum of the absolute differences between coordinates.                                    |
| **Nature of Distance**       | Shortest path in Euclidean space (straight line)                                            | Path along the axes (grid-like path)                                                        |
| **When to Use**              | When the geometry of the space is important and distances are measured in straight lines.   | When movement is restricted to orthogonal directions (e.g., city blocks, grid-based paths). |
| **Computational Complexity** | Generally higher due to the square root calculation                                         | Generally lower, only requires addition and subtraction                                     |
| **Sensitive to Outliers**    | More sensitive to outliers due to squaring of differences                                   | Less sensitive to outliers due to linear summation                                          |
| **Use Cases**                | Physical distances, Euclidean spaces, clustering (like K-means), nearest neighbor searches. | Grid-based problems, certain clustering methods, robust to outliers.                        |
| **Examples**                 | Measuring the distance between two points in a plane.                                       | Calculating the distance between two points in a city grid (taxicab geometry).              |

The R2 score is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It is also known as the coefficient of determination. The formula for R2 is:

R2=1−SStot​SSres​​

where:

- SSres​ is the sum of the squared residuals (the differences between the observed and predicted values).
- SStot​ is the total sum of squares (the differences between the observed values and the mean of the observed values).

### Interpretation of R2 Score

- R2=1: Perfect model; the predictions perfectly fit the actual data.
- R2=0: The model does not explain any of the variance in the target variable; it is as good as using the mean value of the target variable for all predictions.
- R2<0: The model performs worse than simply predicting the mean of the target variable. In this case, the sum of squared residuals is greater than the total sum of squares, indicating poor predictive performance.

### Why R2 Can Be Negative

An R2 score can be negative when the model's predictions are worse than a horizontal line representing the mean of the target variable. This can happen due to:

1. **Overfitting**: The model is too complex and is capturing noise in the training data rather than the underlying trend.
2. **Underfitting**: The model is too simple and fails to capture the underlying trend in the data.
3. **Inappropriate Model**: The chosen model might not be suitable for the data.
4. **Incorrect Feature Engineering**: Features used in the model might not be informative or might be poorly transformed.

--------------

# Correlation

Correlation measures the relationship between two variables. It quantifies the degree to which two variables are related and the direction of that relationship. Here are some key points about correlation:

### Types of Correlation

1. **Positive Correlation**: Both variables move in the same direction. If one increases, the other also increases.
2. **Negative Correlation**: Variables move in opposite directions. If one increases, the other decreases.
3. **No Correlation**: No discernible relationship between the variables.

### Types of Correlation Coefficients

1. **Pearson Correlation Coefficient (r)**: Measures the linear relationship between two continuous variables. It ranges from -1 to 1.
   
   - \( $r$= 1 \): Perfect positive linear relationship.
   - \( $r$ = -1 \): Perfect negative linear relationship.
   - \( $r$ = 0 \): No linear relationship.

2. **Spearman's Rank Correlation Coefficient (\(\rho\))**: Measures the monotonic relationship between two variables. It assesses how well the relationship between two variables can be described using a monotonic function.
   
   - \( $r$ = 1 \): Perfect positive monotonic relationship.
   - \( $r$= -1 \): Perfect negative monotonic relationship.
   - \( $r$= 0 \): No monotonic relationship.

3. **Kendall's Tau (\(\tau\))**: Measures the ordinal association between two variables.

### Calculating Pearson Correlation in Python

Here's how you can calculate the Pearson correlation coefficient using Python's `pandas` and `numpy` libraries:

#### Using `pandas`

```python
import pandas as pd

# Creating a sample dataframe
data = {'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [5, 4, 3, 2, 1]}
df = pd.DataFrame(data)

# Calculating Pearson correlation matrix
correlation_matrix = df.corr(method='pearson')
print(correlation_matrix)
```

#### Using `numpy`

```python
import numpy as np

# Creating sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Calculating Pearson correlation coefficient
correlation_coefficient = np.corrcoef(x, y)[0, 1]
print(f"Pearson correlation coefficient: {correlation_coefficient}")
```

### Visualizing Correlation

You can visualize the correlation between variables using a heatmap:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Creating a sample dataframe
data = {'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [5, 4, 3, 2, 1]}
df = pd.DataFrame(data)

# Calculating the correlation matrix
correlation_matrix = df.corr()

# Plotting the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
```

----------------

![](/home/moaaz/snap/marktext/9/.config/marktext/images/2024-07-01-14-57-07-image.png)

*Standard correlation coefficient of various datasets (source: Wikipedia;
public domain image)*

--------

# Scikit-Learn Design

Scikit-Learn’s API is remarkably well designed. These are the main design principles
Consistency
All objects share a consistent and simple interface:

### Estimators

Any object that can estimate some parameters based on a dataset is called
an estimator (e.g., a SimpleImputer is an estimator). The estimation itself is
performed by the `fit()`method, and it takes a dataset as a parameter, or two
for supervised learning algorithms—the second dataset contains the labels.
Any other parameter needed to guide the estimation process is considered a
hyperparameter (such as a SimpleImputer’s strategy), and it must be set as
an instance variable (generally via a constructor parameter).

### Transformers

Some estimators (such as a **SimpleImputer**) can also transform a dataset;
these are called transformers. Once again, the API is simple: the transforma‐
tion is performed by the `transform()` method with the dataset to transform
as a parameter. It returns the transformed dataset. This transformation gen‐
erally relies on the learned parameters, as is the case for a **SimpleImputer**.
All transformers also have a convenience method called `fit_transform()`,
which is equivalent to calling `fit()` and then `transform()` (but sometimes
`fit_transform()` is optimized and runs much faster).

### Predictors

Finally, some estimators, given a dataset, are capable of making predictions;
they are called predictors. For example, the **LinearRegression** model in
the previous chapter was a predictor: given a country’s GDP per capita, it
predicted life satisfaction. A predictor has a `predict()` method that takes a
dataset of new instances and returns a dataset of corresponding predictions.
It also has a `score()` method that measures the quality of the predictions,
given a test set (and the corresponding labels, in the case of supervised
learning algorithms).10

### Inspection

All the estimator’s **hyperparameters** are accessible directly via public instance
variables (e.g., imputer.strategy), and all the estimator’s learned parameters
are accessible via public instance variables with an underscore suffix (e.g.,
imputer.statistics_).

### Nonproliferation of classes

Datasets are represented as **NumPy** arrays or **SciPy** sparse matrices, instead of
homemade classes. Hyperparameters are just regular Python strings or numbers.

### Composition

Existing building blocks are reused as much as possible. For example, it is easy to
create a Pipeline estimator from an arbitrary sequence of transformers followed
by a final estimator.

### Sensible defaults

**Scikit-Learn** provides reasonable <mark>default values</mark> for most parameters, making it
easy to quickly create a baseline working system.
