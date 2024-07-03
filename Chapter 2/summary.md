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

------------------

# Data Preprocessing

1. Handling missing values.
2. Encoding categorical variables.
3. Scaling numerical features.
4. Splitting the data into training and testing sets.

### Handling Missing Values

1. **Removing Rows with Missing Values**:
   
   - This approach is useful if you have a small number of missing values and removing them won't significantly reduce your dataset size.

2. **Removing Columns with Missing Values**:
   
   - This approach is useful if certain columns have a high percentage of missing values and are not critical for your analysis.

3. **Imputing Missing Values**:
   
   - Numerical columns: Impute using the mean, median, or a constant value.
   - Categorical columns: Impute using the most frequent value or a constant value.

4. **Advanced Imputation Techniques**:
   
   - KNN imputation or predictive modeling to fill in missing values based on other available data.

```python
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

# Load the dataset
file_path = '/mnt/data/housing.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values before handling:\n", missing_values)

# 1. Removing rows with missing values
df_dropped_rows = df.dropna()
print("Shape after dropping rows with missing values:", df_dropped_rows.shape)

# 2. Removing columns with missing values
df_dropped_columns = df.dropna(axis=1)
print("Shape after dropping columns with missing values:", df_dropped_columns.shape)

# 3. Imputing missing values
# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Impute numerical columns with the median
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categorical columns with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("Missing values after simple imputation:\n", df.isnull().sum())

# 4. Advanced Imputation Technique: KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = knn_imputer.fit_transform(df)

# Convert back to DataFrame
df_knn_imputed = pd.DataFrame(df_knn_imputed, columns=df.columns)
print("Missing values after KNN imputation:\n", df_knn_imputed.isnull().sum())

# Choose the method that best suits your dataset
df_final = df_knn_imputed


target_column = 'target' 
X = df_final.drop(columns=target_column)
y = df_final[target_column]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

### Encoding categorical variables

1. **Label Encoding**

Label encoding assigns a unique integer to each category. It's suitable for ordinal categorical variables (where the categories have a meaningful order).

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Example DataFrame
data = {'category': ['A', 'B', 'A', 'C', 'B']}
df = pd.DataFrame(data)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical column
df['category_encoded'] = label_encoder.fit_transform(df['category'])

print(df)
```

Output:

```
  category  category_encoded
0        A                 0
1        B                 1
2        A                 0
3        C                 2
4        B                 1
```

2. **One-Hot Encoding**

One-hot encoding creates binary columns for each category. It's suitable for nominal categorical variables (where categories don't have an inherent order).

```python
from sklearn.preprocessing import OneHotEncoder

# Example DataFrame
data = {'category': ['A', 'B', 'A', 'C', 'B']}
df = pd.DataFrame(data)

# Initialize OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)

# Encode categorical column
onehot_encoded = onehot_encoder.fit_transform(df[['category']])

# Create DataFrame from encoded data
df_encoded = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['category']))

# Concatenate with original DataFrame
df = pd.concat([df, df_encoded], axis=1)

print(df)
```

Output:

```
  category    x0_A  x0_B  x0_C
0        A   1.0    0.0   0.0
1        B   0.0    1.0   0.0
2        A   1.0    0.0   0.0
3        C   0.0    0.0   1.0
4        B   0.0    1.0   0.0
```

3. **Ordinal Encoding**

Ordinal encoding maps categorical values to integers based on a specified order. It's useful for categorical variables with a clear ordering.

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Example DataFrame
data = {'size': ['S', 'M', 'L', 'M', 'XL']}
df = pd.DataFrame(data)

# Define the order of categories
size_order = ['S', 'M', 'L', 'XL']

# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[size_order])

# Encode categorical column
df['size_encoded'] = ordinal_encoder.fit_transform(df[['size']])

print(df)
```

Output:

```
  size  size_encoded
0    S           0.0
1    M           1.0
2    L           2.0
3    M           1.0
4   XL           3.0
```

- **Label Encoding**: Converts each category into a numerical label. Suitable for ordinal data.
- **One-Hot Encoding**: Creates binary columns for each category. Suitable for nominal data.
- **Ordinal Encoding**: Maps categories to integers based on a specified order. Suitable for ordinal data with a clear order.

### Scaling numerical features.

Scaling numerical features is an essential preprocessing step in machine learning to ensure that all features contribute equally to the model training process. Here are common methods used for scaling numerical features:

### 1. Min-Max Scaling (Normalization)

Min-max scaling transforms features to a range between 0 and 1. It is defined by the formula:
![](/home/moaaz/snap/marktext/9/.config/marktext/images/2024-07-03-15-31-33-image.png)

### 2. Standardization (Z-score Normalization)

Standardization transforms features to have a mean of 0 and a standard deviation of 1. It is defined
![](/home/moaaz/snap/marktext/9/.config/marktext/images/2024-07-03-15-32-40-image.png)

### 3. Robust Scaling

Robust scaling is useful when the dataset contains outliers. It scales features using the interquartile range (IQR) instead of the minimum and maximum values:
![](/home/moaaz/snap/marktext/9/.config/marktext/images/2024-07-03-15-33-26-image.png)

### Implementation in Python (using scikit-learn)

Here's how you can implement these scaling techniques using scikit-learn:

#### Min-Max Scaling

```python
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)
```

#### Standardization

```python
from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)
```

#### Robust Scaling

```python
from sklearn.preprocessing import RobustScaler

# Initialize RobustScaler
scaler = RobustScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)
```

### Choosing a Scaling Method

- **Min-Max Scaling:** Use when you know the distribution of your data is uniform or when you want your features to be on a similar scale.
- **Standardization:** Use when your data follows a normal distribution, and you want to maintain interpretability of feature importance or when algorithms like SVMs or K-Means clustering are used.
- **Robust Scaling:** Use when dealing with outliers or when the distribution of your data is not normal.

### Applying Scaling to Specific Columns

If you want to scale only specific columns (features) in your DataFrame `df`, you can extract those columns and scale them accordingly:

```python
columns_to_scale = ['feature1', 'feature2', 'feature3']
X_scaled = df[columns_to_scale]

# Apply scaling
X_scaled = scaler.fit_transform(X_scaled)
```

### Splitting the data into training and testing sets.

#### Using scikit-learn (`train_test_split`)

Scikit-learn provides a convenient function `train_test_split` to split your dataset into training and testing sets. This function shuffles and splits your data into two subsets according to specified proportions.

#### Syntax

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Parameters

- **X:** Features (data matrix).
- **y:** Target variables (labels).
- **test_size:** float or int, default=0.25. If float, should be between 0.0 and 1.0 and represents the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
- **train_size:** float or int, default=None. If provided, overrides `test_size`. Represents the proportion of the dataset to include in the train split.
- **random_state:** int or RandomState instance, default=None. Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

#### Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset into a DataFrame (assuming 'df' is your DataFrame)
# Example:
# X = df[['feature1', 'feature2', ...]]
# y = df['target_variable']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
```

#### Notes

- **X_train, X_test:** The feature sets for training and testing.
- **y_train, y_test:** The corresponding target sets for training and testing.
- **test_size:** You can adjust this parameter to change the proportion of the dataset allocated for testing.
- **random_state:** Setting a value here ensures that the data split is reproducible, which is important for getting consistent results when you run your code multiple times.

### Best Practices

- **Random Shuffling:** Always shuffle your data before splitting to ensure that the distribution of classes or patterns is consistent across training and testing sets.
- **Stratification:** For classification tasks, consider using `stratify=y` in `train_test_split` to maintain the same class distribution in both training and testing sets.
