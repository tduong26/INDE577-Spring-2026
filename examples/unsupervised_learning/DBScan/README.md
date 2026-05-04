# DBSCAN

## Overview

Density-Based Spatial Clustering of Applications with Noise (**DBSCAN**)
is an **unsupervised clustering** algorithm that groups observations
according to **local density** rather than distance to a central mean.

Unlike centroid-based methods such as K-Means, DBSCAN does not require
the number of clusters to be specified in advance, can recover clusters
of arbitrary shape, and naturally identifies outliers as noise.

This notebook demonstrates DBSCAN **from scratch** using the `rice_ml`
package and applies it to the canonical **two-moons** dataset to
illustrate both the density-based intuition and the practical role of
hyperparameter selection.

---

## Key Idea

Given a dataset with continuous features, DBSCAN:

- Defines clusters as **dense regions** of points connected by
  neighborhood relationships
- Identifies isolated observations as **noise** rather than forcing
  them into a cluster
- Discovers clusters of **arbitrary shape**, not just spherical groups
- Determines the number of clusters automatically from the data

This makes DBSCAN especially useful when clusters have irregular shapes
or when the dataset contains outliers.

---

## Mathematical Intuition

DBSCAN is controlled by two main hyperparameters:

- **$\varepsilon$ (eps)**: the radius used to define a neighborhood  
- **`min_samples`**: the minimum number of nearby points required to
  declare a dense region  

We write `eps` and `min_samples` in code, and use $\varepsilon$ and
$m$ in equations.

### $\varepsilon$-Neighborhood

For a point $x$ in a dataset $D$, the $\varepsilon$-neighborhood is the
set of all points within distance $\varepsilon$ of $x$:

$$
N_\varepsilon(x) = \{\, y \in D \mid d(x,y) \le \varepsilon \,\}
$$

where $d(x, y)$ denotes the Euclidean distance between $x$ and $y$.

### Core Point

A point $x$ is a **core point** if its $\varepsilon$-neighborhood
contains at least $m$ points (where $m = $ `min_samples`):

$$
|N_\varepsilon(x)| \ge m
$$

### Border and Noise Points

- A **border point** lies inside the neighborhood of a core point but
  does not itself satisfy the density requirement  
- A **noise point** is not density-reachable from any core point and is
  labeled as `-1`  

DBSCAN builds each cluster by starting from a core point and repeatedly
adding all points that are density-reachable from it.

---

## Dataset

The notebook uses the **`make_moons`** synthetic dataset from
scikit-learn, the canonical demonstration dataset for density-based
clustering. The dataset produces two **interlocking crescent-shaped
clusters** — exactly the kind of non-convex structure that centroid-based
methods like K-Means cannot recover correctly, but that DBSCAN handles
naturally.

Dataset characteristics:

- Source: `sklearn.datasets.make_moons` (synthetic)  
- 300 total observations  
- 2 numeric features (x, y coordinates)  
- 2 true clusters (used only for visual comparison, not training)  
- Noise level: 0.08 (mild Gaussian perturbation that creates a few
  ambiguous boundary points)  
- No missing values  

### Why this dataset works for DBSCAN

- The two clusters are **non-spherical** — K-Means would slice them in
  half down the middle, while DBSCAN follows the density of points
  along the curves.  
- Adding small Gaussian noise creates a realistic scenario with a few
  candidate outliers, which DBSCAN flags as noise.  
- The data is **2-dimensional**, so cluster results can be directly
  visualized without dimensionality reduction.  
- It loads offline through scikit-learn, so the notebook runs reliably
  without depending on a network connection.  

---

## Exploratory Analysis

For DBSCAN, exploratory analysis is mainly about understanding the
**shape, spacing, and density** of the observations.

Key questions:

- Do the points form visibly separated groups?  
- Are there boundary observations between groups?  
- Is standardization needed before clustering?  
- What values of `eps` and `min_samples` seem reasonable?  

Key observations:

- The two crescents are clearly visible in raw feature space  
- A few points sit in the gap between the two moons due to added noise  
- Both features are already on comparable numeric scales, but
  standardization is still applied as a best-practice step  

Because DBSCAN depends entirely on distance, feature standardization is
included in the workflow even when its effect is small on synthetic
data.

---

## Preprocessing

The two features are standardized to zero mean and unit variance:

$$
X_{\text{std}} = \frac{X - \mu}{\sigma}
$$

After standardization, both features contribute fairly to Euclidean
distance calculations. A second scatter plot in standardized space
confirms that the visible crescent structure is preserved while the
axes are now on comparable scales.

---

## Model Training

DBSCAN is fit on the standardized feature space with:

- `eps = 0.2`  
- `min_samples = 5`  

These values are chosen so that points within the same crescent are
density-connected, while the gap between the two crescents is wider
than `eps` — letting DBSCAN identify them as separate clusters.

The output of `fit_predict` is a label vector where:

- Nonnegative integers (`0`, `1`) correspond to the two identified
  clusters  
- The label `-1` marks noise points that are not part of any dense
  region  

A summary reports the count of points in each label and confirms the
total number of clusters discovered.

---

## Visualizing the Clusters

The notebook produces two scatter plots side by side:

- **Left**: DBSCAN's clustering result, with each point colored by its
  assigned cluster (or marked as noise)  
- **Right**: The true cluster labels from `make_moons`, shown for
  comparison  

The plots reveal:

- Both crescents are correctly identified as separate density-connected
  clusters  
- The shape of each cluster follows the curve of the moon, not a
  circular blob  
- A small number of points in the gap between the moons may be flagged
  as noise, reflecting the Gaussian noise added during data generation  

This behavior is fundamentally different from K-Means, which would slice
the two crescents in half down the middle and produce two clusters that
don't match the true shapes at all.

---

## Interpreting the Clusters

After fitting DBSCAN, each observation is either part of a
density-connected cluster or marked as noise.

When reading the results:

- Points in the same cluster are connected through chains of dense
  neighborhoods, which is why DBSCAN can recover curved shapes  
- Cluster shapes can be arbitrary — DBSCAN does not assume spherical
  groups  
- Noise points represent genuinely isolated observations, not weak
  cluster members  
- Different `eps` values produce different cluster structures: too
  small fragments the moons into many small clusters, too large merges
  them into one  

Because this is an unsupervised method, the cluster labels are
data-driven groupings, not predetermined categories.

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
- Can recover non-spherical structure better than centroid-based
  methods  

### Limitations

- Results can be sensitive to the choice of `eps` and `min_samples`  
- Performance may decrease when clusters have very different densities  
- In higher dimensions, distance measures can become less informative  
- Some datasets may require trial and error to find useful
  hyperparameters  

---

## Conclusion

This notebook demonstrates the canonical DBSCAN use case: clusters
defined by **shape and density** rather than by proximity to a center.

Key takeaways:

- DBSCAN groups points based on local density rather than distance to
  a centroid  
- The two-moons dataset is the textbook example of where density-based
  clustering wins over K-Means — interlocking crescents that no
  centroid-based method can correctly separate  
- Standardization is included as a best-practice step even when its
  effect is small on synthetic data  
- The number of clusters emerges from the data rather than being
  chosen in advance  
- Hyperparameter selection (`eps`, `min_samples`) is a meaningful part
  of the modeling process  

DBSCAN remains a core tool for unsupervised analysis, particularly when
clusters have irregular shapes or when outlier detection is part of the
goal. In this project, DBSCAN complements the K-Means and PCA notebooks
as an additional unsupervised learning example, alongside the supervised
methods (Perceptron, MLP, Decision Tree, KNN, Ensemble Methods)
elsewhere in the repository.