# K-Means Clustering

## Overview

K-Means is an **unsupervised clustering** algorithm that partitions a dataset into `K` groups by assigning each observation to the nearest centroid and iteratively updating those centroids.

Unlike supervised methods, K-Means does not use target labels. Instead, it discovers natural groupings based purely on similarity in the feature space.

This notebook demonstrates K-Means **from scratch** using the `rice_ml` package and applies it to a real-world dataset to illustrate both the algorithm and its practical interpretation.

---

## Key Idea

Given a dataset with multiple continuous features, K-Means:

- Places `K` centroids in the feature space
- Assigns each point to the **nearest centroid** under Euclidean distance
- Updates each centroid as the **mean of its assigned points**
- Repeats until cluster assignments stop changing

The result is a partition of the data into `K` groups whose members are similar to each other and dissimilar to members of other groups.

---

## K-Means Objective Function

K-Means minimizes the total **within-cluster sum of squared distances**:

$$
\sum_{i=1}^{n} \left\| x_i - \mu_{c(i)} \right\|^2
$$

Where:
- $x_i$ is a data point
- $\mu_{c(i)}$ is the centroid of the cluster assigned to $x_i$
- $c(i)$ denotes the cluster index for point $i$

This quantity is also called **inertia** and serves as the primary measure of clustering quality.

---

## Algorithm Steps

The K-Means algorithm proceeds iteratively:

1. **Initialize** `K` cluster centroids
2. **Assign** each data point to the nearest centroid
3. **Update** each centroid as the mean of its assigned points
4. **Repeat** until convergence or until the maximum number of iterations is reached

Convergence is reached when centroid positions stop changing or fall below a tolerance threshold.

---

## Dataset

The notebook uses the **Wine** dataset from the UCI Machine Learning Repository, which contains physicochemical measurements of wines.

Dataset characteristics:

- 13 continuous numeric features (alcohol, malic acid, ash, magnesium, flavanoids, color intensity, hue, proline, etc.)
- Features vary widely in scale  
- No missing values  
- Contains class labels, but they are **not used** since K-Means is unsupervised  

These properties make the dataset well-suited for demonstrating K-Means.

---

## Exploratory Analysis

Before applying K-Means, exploratory analysis examines:

- True class distribution (for reference only — not used in training)  
- Feature scale comparison (boxplot of all 13 features)  

Key observations include:

- Features differ significantly in magnitude  
- Some features (such as proline) span far larger numeric ranges than others  
- These scale differences would dominate Euclidean distance calculations if left unaddressed  

Because K-Means is distance-based, **feature standardization** is essential prior to clustering.

---

## Preprocessing

All features are standardized to zero mean and unit variance:

$$
X_{\text{std}} = \frac{X - \mu}{\sigma}
$$

This ensures that every feature contributes equally to distance computations and improves the stability of the algorithm.

---

## Model Training

K-Means is fit on the standardized data with the following configuration:

- `n_clusters = 3`  
- `max_iter = 300`  
- `tol = 1e-4`  
- `random_state = 42`  

After training, three outputs are inspected:

- **Labels** — the cluster assignment for each observation  
- **Cluster centers** — the mean position of each cluster in feature space  
- **Inertia** — total within-cluster sum of squared distances  

---

## Elbow Method

The elbow method helps determine a reasonable number of clusters by plotting inertia as a function of `K`.

Important observations:

- Inertia always decreases as `K` increases  
- A sharp slowdown ("elbow") indicates diminishing returns from adding more clusters  
- The elbow location suggests a good trade-off between model complexity and clustering quality  

In this notebook, K-Means is fit for `K = 1` through `K = 10`. The curve begins to flatten around `K = 3`, suggesting that three clusters capture most of the structure in the data.

---

## Low-Dimensional Visualization

Because the dataset has 13 features, direct visualization is not possible. Principal Component Analysis (PCA) is used to project the data into two dimensions for display.

The 2D scatter plot shows:

- Each observation colored by its assigned cluster  
- Cluster centroids overlaid as red markers in PCA space  

Important note:

- PCA is used **only for visualization**  
- Clustering is performed in the original 13-dimensional standardized feature space  

---

## Interpreting the Clusters

After fitting K-Means, each observation belongs to the cluster with the nearest centroid.

When reading the results:

- Points in the same cluster are similar under Euclidean distance  
- Cluster centers summarize the average location of each group  
- Separation in the PCA plot suggests that K-Means has identified distinct patterns in the data  
- Overlap between groups may indicate that the clusters are not strongly separated in the original feature space  

Because this is an unsupervised method, clusters do not automatically come with real-world labels. They should be interpreted as data-driven groupings, not as guaranteed categories.

---

## When to Use K-Means

K-Means is well-suited for:

- Exploratory grouping of unlabeled data  
- Customer or market segmentation  
- Image color quantization  
- Quick baselines for clustering tasks  

K-Means is **not ideal** when:

- The number of clusters is genuinely unknown and difficult to estimate  
- Clusters have irregular shapes or very different sizes  
- The data contains many outliers  
- Variables are categorical rather than continuous  

---

## Limitations

Despite its usefulness, K-Means has important limitations:

- The number of clusters `K` must be chosen in advance  
- Results depend on the initialization of centroids  
- It assumes roughly spherical, similarly sized clusters  
- It is sensitive to outliers and feature scale  
- It may perform poorly when clusters overlap or have non-convex shapes  

---

## Conclusion

This notebook demonstrates K-Means as a foundational unsupervised learning technique.

Key takeaways:

- K-Means partitions data by minimizing within-cluster squared distance  
- Feature standardization is essential for meaningful results  
- The elbow method provides a practical guide for choosing `K`  
- PCA is useful for visualizing the resulting clusters  

K-Means remains a core tool in modern machine learning pipelines, particularly as a first-pass exploratory method and as a baseline for more advanced clustering algorithms. In this project, K-Means complements the PCA notebook as the second unsupervised learning example, alongside the supervised methods (Perceptron, MLP, Decision Tree, KNN, Ensemble Methods) elsewhere in the repository.