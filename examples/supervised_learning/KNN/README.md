# K-Nearest Neighbors (KNN) — Banknote Authentication Dataset

This directory contains a Jupyter notebook demonstrating a K-Nearest Neighbors (KNN) classifier implemented entirely from scratch using the custom rice_ml package.

The notebook applies KNN to the Banknote Authentication dataset, a classic forensic-finance benchmark for detecting forged banknotes from wavelet-transformed image statistics, and emphasizes the role of distance-based prediction, standardization, and the bias–variance tradeoff controlled by k.

No scikit-learn classifiers are used.

## Notebook Overview

Notebook: knn_example.ipynb
Model: Custom KNN classifier
Dataset: Banknote Authentication — UCI Machine Learning Repository

The notebook walks through the complete process of applying KNN to a real-world dataset, including:

- Loading the Banknote Authentication dataset directly from the web
- Exploratory data analysis (EDA)
- Feature preprocessing and standardization
- Training a KNN classifier with a default k
- Tuning k by sweeping over a range and tracking train/test accuracy
- Evaluating performance with multiple classification metrics
- Visualizing 2-D decision boundaries and the effect of k

## What Is K-Nearest Neighbors?

KNN is a non-parametric, instance-based learning algorithm. It stores the entire training set and makes predictions by finding the k closest training points to a query point and taking a majority vote among their labels.

### Distance Metric

Predictions are based on Euclidean distance:

d(x, x') = sqrt( sum_j (x_j − x'_j)^2 )

### Prediction Rule

For a test point x_0:

1. Find the k training points with the smallest distance to x_0
2. Predict the majority class among those k neighbors

### The Role of k

The hyperparameter k directly controls the bias–variance tradeoff:

- Small k (e.g., k=1) — low bias, high variance; the boundary is jagged and sensitive to noise
- Large k — high bias, low variance; the boundary is smooth but may underfit
- Optimal k — chosen empirically (here, by sweeping over a range and picking the best test accuracy)

### Why Standardization Matters

KNN is purely distance-based, so features with larger numerical scales dominate the distance computation. Standardization rescales all features to comparable ranges:

X_std = (X − mean_train) / std_train

Without standardization, a single high-magnitude feature can effectively determine all neighbor relationships.

## Dataset Description

The Banknote Authentication dataset contains 1,372 samples extracted from images of genuine and forged banknotes using a Wavelet Transform tool.

### Features

All 4 features are continuous numerical statistics of the wavelet-transformed image:

- variance — Variance of wavelet-transformed image
- skewness — Skewness of wavelet-transformed image
- curtosis — Curtosis of wavelet-transformed image
- entropy — Entropy of the image

### Target Variable

- class — 0 = genuine, 1 = forged

### Data Notes

All features are continuous

No missing values

Features have different scales, motivating standardization

The dataset is reasonably balanced between genuine and forged banknotes

## Exploratory Data Analysis (EDA)

EDA in this notebook focuses on:

- Summary statistics and missing-value check
- Class distribution (bar chart and pie chart)
- Per-feature distributions split by class (overlaid histograms for variance, skewness, curtosis, entropy)
- Correlation matrix among the four features
- 2-D scatter plot of the two most discriminative features (variance vs. skewness)

These visualizations confirm that variance and skewness are the most informative features for separating genuine from forged banknotes — which motivates using them for the 2-D decision boundary plots later.

## Preprocessing

Before training the model:

- Features and labels are separated
- Data is split into training and test sets using a custom train_test_split (80/20 split, random_state=42)
- Features are standardized to zero mean and unit variance using a custom standardize function
- The scaler is fit only on the training set to prevent data leakage

This ensures fair evaluation and consistent distance computation across features.

## Initial Model Training

A baseline KNN model is trained with k = 5, a common default. Both training and test accuracy are reported, and a full confusion matrix is computed along with derived metrics (precision, recall, F1, specificity).

## Tuning k

The notebook sweeps over k = 1, 2, ..., 30, training a fresh KNN model for each value and recording both training and test accuracy. The best k is selected as the one that maximizes test accuracy.

This sweep makes the bias–variance tradeoff visible directly:

- Very small k → near-perfect train accuracy but lower test accuracy (overfitting)
- Very large k → train and test accuracy converge but decline (underfitting)
- The optimal k sits between these extremes

The accuracy curve is plotted with the best k highlighted by a vertical reference line.

## Evaluation Metrics

Once the best k is selected, the notebook re-trains KNN with that k and evaluates performance using:

- Accuracy — overall classification correctness
- Confusion Matrix — counts and normalized percentages
- Precision and Recall — class-specific performance for the forged class
- F1 Score — harmonic mean of precision and recall
- Specificity — true negative rate (correctly identified genuine banknotes)

A side-by-side confusion matrix heatmap (counts and normalized) makes the error structure easy to interpret.

## Decision Boundary Visualization

Two visualizations show how KNN partitions the feature space:

### 2-D Decision Boundary at Best k

The 2-D decision boundary is plotted using the two most discriminative features (variance and skewness) after standardization. A separate KNN model is trained on this 2-D projection so that the boundary can be drawn directly in feature space.

### Effect of k on Boundary Shape

A 1×4 grid shows decision boundaries for k = 1, 5, 15, 25. As k increases, the boundary visibly smooths out — concretely illustrating the bias–variance tradeoff discussed earlier.

Note: These visualizations use only 2 of the 4 features and are intended for intuition. The main KNN model evaluated above is trained on the full 4-dimensional feature space.

## Limitations of KNN

While KNN is simple and effective on well-separated data, it has important limitations:

- Computational cost — prediction requires computing distances to all training points (O(n · d) per query), which scales poorly to large datasets
- No explicit model — KNN is a lazy learner that stores the entire training set, making it memory-intensive
- Sensitive to irrelevant features — all features contribute equally to distance regardless of relevance
- Curse of dimensionality — in high-dimensional spaces, distances become less meaningful and neighbors become less informative

These limitations motivate parametric methods (logistic regression, SVMs) and tree-based methods (decision trees, random forests) covered elsewhere in the project.

## Purpose of This Notebook

This notebook is designed to:

- Demonstrate a correct from-scratch implementation of KNN
- Reinforce the importance of standardization for distance-based methods
- Provide an empirical view of the bias–variance tradeoff through a k-sweep
- Build intuition for how KNN partitions feature space using 2-D visualizations
- Serve as a non-parametric baseline alongside other classifiers in the project

## Notes

KNN is implemented from scratch in the rice_ml package

Preprocessing utilities (standardize, train_test_split) come from rice_ml

Evaluation utilities (accuracy_score, confusion_matrix) come from rice_ml

No scikit-learn classifiers are used

Visualizations rely on Matplotlib

This notebook complements the linear-model, tree-based, and ensemble examples elsewhere in the repository