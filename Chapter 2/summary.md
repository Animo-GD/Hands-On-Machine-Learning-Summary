d# Chapter 2

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




