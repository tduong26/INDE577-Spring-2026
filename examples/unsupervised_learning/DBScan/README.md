# DBSCAN 

## Overview

Density-Based Spatial Clustering of Applications with Noise (**DBSCAN**) is an **unsupervised clustering** algorithm that groups observations according to **local density** rather than distance to a central mean.

Unlike centroid-based methods such as K-Means, DBSCAN does not require the number of clusters to be specified in advance and can naturally identify outliers as noise.

This notebook demonstrates DBSCAN **from scratch** using the `rice_ml` package and applies it to a real-world dataset to illustrate both the density-based intuition and the practical role of hyperparameter selection.

---

## Key Idea

Given a dataset with continuous features, DBSCAN:

- Defines clusters as **dense regions** of points connected by neighborhood relationships
- Identifies isolated observations as **noise** rather than forcing them into a cluster
- Discovers clusters of **arbitrary shape**, not just spherical groups
- Determines the number of clusters automatically from the data

This makes DBSCAN especially useful when clusters have irregular shapes or when the dataset contains outliers.

---

## Mathematical Intuition

DBSCAN is controlled by two main hyperparameters:

- **$\varepsilon$ (eps)**: the radius used to define a neighborhood  
- **`min_samples`**: the minimum number of nearby points required to declare a dense region  

### $\varepsilon$-Neighborhood

For a point $x$, the $\varepsilon$-neighborhood is:

$$
N_\varepsilon(x) = \{ y \mid \text{distance}(x,y) \le \varepsilon \}
$$

### Core Point

A point $x$ is a **core point** if its neighborhood contains at least `min_samples` points:

$$
|N_\varepsilon(x)| \ge \text{min\_samples}
$$

### Border and Noise Points

- A **border point** lies inside the neighborhood of a core point but does not itself satisfy the density requirement  
- A **noise point** is not density-reachable from any core point and is labeled as `-1`  

DBSCAN builds each cluster by starting from a core point and repeatedly adding all points that are density-reachable from it.

---

## Dataset

The notebook uses the classic **Iris** dataset, loaded directly from a public web URL so that no local file is required.

Dataset characteristics:

- 150 total observations  
- 4 numeric features in the original dataset  
- 2 features used in this notebook: `Petal.Length` and `Petal.Width`  
- 3 underlying species (used only for context, not in clustering)  
- No missing values  

These two petal measurements show clear group structure in two dimensions, which makes them well-suited for visualizing DBSCAN.

---

## Exploratory Analysis

For DBSCAN, exploratory analysis is mainly about understanding the **shape, spacing, and density** of the observations.

Key questions:

- Do the points form visibly separated groups?  
- Are there boundary observations between groups?  
- Is standardization needed before clustering?  
- What values of `eps` and `min_samples` seem reasonable?  

Key observations:

- Petal Length and Petal Width are measured on different numeric scales  
- One group is clearly separated from the others in raw feature space  
- A few boundary points sit between dense regions  

Because DBSCAN depends entirely on distance, **feature standardization** is essential prior to clustering.

---

## Preprocessing

The two petal features are standardized to zero mean and unit variance:

$$
X_{\text{std}} = \frac{X - \mu}{\sigma}
$$

After standardization, both features contribute fairly to Euclidean distance calculations. A second scatter plot in standardized space confirms that the visible group structure is preserved while the axes are now on comparable scales.

---

## Model Training

DBSCAN is fit on the standardized feature space with:

- `eps = 0.35`  
- `min_samples = 5`  

These values are chosen to give a clear illustration on the standardized Iris features. Because DBSCAN does not require a target number of clusters, the algorithm decides the number of groups itself based on density.

The output of `fit_predict` is a label vector where:

- Nonnegative integers (`0`, `1`, `2`, ...) correspond to identified clusters  
- The label `-1` marks noise points that are not part of any dense region  

A summary table reports the count of points in each label, including any noise points.

---

## Visualizing the Clusters

A scatter plot in standardized petal space shows each observation colored by its DBSCAN-assigned cluster.

The plot reveals:

- A dense, compact group identified very clearly  
- Boundary observations attached to whichever dense region they connect to  
- A few isolated points labeled as noise if they lack enough neighbors  

This behavior is fundamentally different from K-Means, which always assigns every point to a cluster regardless of how isolated the point is.

---

## Interpreting the Clusters

After fitting DBSCAN, each observation is either part of a density-connected cluster or marked as noise.

When reading the results:

- Points in the same cluster are connected through chains of dense neighborhoods  
- Cluster shapes can be arbitrary — DBSCAN does not assume spherical groups  
- Noise points represent genuinely isolated observations rather than weak cluster members  
- Different `eps` values produce different cluster structures, so hyperparameter choice is part of the modeling decision  

Because this is an unsupervised method, clusters do not automatically correspond to real-world species labels. They should be interpreted as data-driven groupings based on density.

---

## When to Use DBSCAN

DBSCAN is well-suited for:

- Discovering clusters of arbitrary shape  
- Identifying outliers naturally as part of clustering  
- Datasets where the number of clusters is unknown  
- Situations where points should not be forced into a group  

DBSCAN is **not ideal** when:

- Clusters have very different densities  
- The dataset is high-dimensional (distance becomes less informative)  
- Useful values of `eps` and `min_samples` are difficult to find  
- Variables are categorical rather than continuous  

---

## Strengths and Limitations

### Strengths

- Works well for density-connected groups  
- Can identify outliers naturally  
- Does not require pre-specifying the number of clusters  
- Can recover non-spherical structure better than centroid-based methods  

### Limitations

- Results can be sensitive to the choice of `eps` and `min_samples`  
- Performance may decrease when clusters have very different densities  
- In higher dimensions, distance measures can become less informative  
- Some datasets may require trial and error to find useful hyperparameters  

---

## Conclusion

This notebook demonstrates DBSCAN as a foundational unsupervised learning technique.

Key takeaways:

- DBSCAN groups points based on local density rather than distance to a centroid  
- Standardization is essential because cluster membership is determined entirely from neighborhood distances  
- The number of clusters emerges from the data rather than being chosen in advance  
- Hyperparameter selection (`eps`, `min_samples`) is a meaningful part of the modeling process  

DBSCAN remains a core tool for unsupervised analysis, particularly when clusters have irregular shapes or when outlier detection is part of the goal. In this project, DBSCAN complements the K-Means and PCA notebooks as an additional unsupervised learning example, alongside the supervised methods (Perceptron, MLP, Decision Tree, KNN, Ensemble Methods) elsewhere in the repository.