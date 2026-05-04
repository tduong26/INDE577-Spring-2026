# Decision Tree Classifier — Wine Quality Dataset

This directory contains a Jupyter notebook demonstrating a Decision Tree classifier implemented entirely from scratch using the custom rice_ml package.

The notebook emphasizes both algorithmic intuition and practical application, following a standard supervised learning workflow without reliance on scikit-learn models.

## Notebook Overview

Notebook: decision_tree_example.ipynb
Model: Custom DecisionTree classifier
Dataset: Wine Quality (Red) — UCI Machine Learning Repository

The notebook walks through the complete process of applying a decision tree to a real-world dataset, including:

- Dataset loading and inspection
- Exploratory data analysis (EDA)
- Target construction for binary classification
- Training a decision tree from scratch with depth constraints
- Model evaluation on training and test data
- Visualization of decision regions using PCA

## What Is a Decision Tree?

A Decision Tree is a supervised learning model that predicts outcomes by recursively partitioning the feature space into regions that are increasingly homogeneous with respect to the target variable.

At each internal node, the model applies a rule of the form:

- Is feature_j less than or equal to some threshold?

Each leaf node stores class statistics and predicts the most likely class for observations that fall into that region.

The implementation in this project uses Gini impurity to measure node purity and selects splits greedily to minimize weighted impurity:

G = 1 − sum_c p(c)^2

Where p(c) is the proportion of class c in the node.

- Gini = 0 → pure node
- Gini near 0.5 → mixed classes

At each node, the tree evaluates all possible feature–threshold pairs and chooses the split that minimizes the weighted Gini impurity of the children. This greedy process is repeated recursively until stopping criteria (such as maximum depth) are met.

## Dataset Description

The Wine Quality (Red) dataset consists of 1,599 Portuguese red wines, each described by 11 continuous physicochemical features such as acidity, alcohol content, and sulphates.

The original quality score is an integer between 0 and 10. For this notebook, the target variable is converted into a binary classification problem:

- 1 — good quality wine (quality ≥ 6)
- 0 — lower quality wine (quality < 6)

The dataset contains no missing values, making it well-suited for tree-based modeling.

## Exploratory Data Analysis

The notebook includes exploratory analysis to:

- Examine the distribution of original wine quality scores (bar chart)
- Visualize the distribution of alcohol content (histogram)
- Compare each feature against the binary target (boxplots for all 11 predictors)
- Examine the correlation structure between features (heatmap)

Although decision trees do not require feature scaling, EDA plays an important role in interpreting which splits the tree may learn. Features such as alcohol, volatile acidity, and sulphates appear especially informative.

## Model Training and Evaluation

A depth-constrained decision tree (max_depth=4) is trained to balance model complexity and generalization. The data is split using a custom train_test_split function with a 75/25 split and random_state=42.

The notebook evaluates:

- Training accuracy
- Test accuracy
- Evidence of mild overfitting (gap between train and test accuracy)
- The effect of limiting tree depth

This analysis highlights the bias–variance tradeoff inherent in tree-based models.

## PCA-Based Visualization

Because the dataset has 11 features, direct visualization of decision boundaries is not possible. Principal Component Analysis (PCA) — also implemented in the rice_ml package — is used to project the data into two dimensions for visualization only.

A separate decision tree is trained on the PCA-transformed data to illustrate:

- Axis-aligned decision regions (vertical and horizontal bands in PCA space)
- Approximate class separation along the first principal component
- The geometric behavior of tree-based models
- Train vs. test point consistency in the projected space

The visualization overlays training points and test points (marked separately) on top of the predicted decision regions. This provides qualitative insight but does not represent the exact decision logic of the original 11-dimensional model.

## Purpose of This Notebook

This notebook is designed to:

- Demonstrate a decision tree classifier implemented from first principles
- Reinforce intuition behind recursive partitioning and impurity measures
- Show how decision trees behave on real, noisy data
- Serve as a baseline before more complex ensemble methods (Random Forests, Gradient Boosting)
- Provide a reusable template for tree-based modeling in this project

## Notes

The decision tree is implemented from scratch in the rice_ml package

PCA for visualization is also from the rice_ml package

No scikit-learn decision tree models are used

Visualizations are created using Matplotlib and Seaborn

PCA is used strictly for visualization and interpretation purposes

Mathematical expressions are written using GitHub-compatible LaTeX

This notebook complements the source code and unit tests by providing a clear, interpretable example of decision tree learning in practice.