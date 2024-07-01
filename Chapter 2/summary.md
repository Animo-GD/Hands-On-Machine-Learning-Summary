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
