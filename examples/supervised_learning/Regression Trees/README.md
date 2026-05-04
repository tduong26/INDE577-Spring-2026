# Regression Tree — California Housing Dataset

This directory contains a Jupyter notebook demonstrating a Regression Tree implemented entirely from scratch using the custom rice_ml package.

The notebook applies the regression tree to the California Housing dataset, a classic benchmark for nonlinear regression on tabular data, and emphasizes interpretability, piecewise-constant prediction, and proper diagnostic evaluation.

No scikit-learn regression models are used.

## Notebook Overview

Notebook: regression_trees_example.ipynb
Model: Custom RegressionTree
Dataset: California Housing — 1990 U.S. Census (block-group level)

The notebook walks through the complete process of applying a regression tree to a real-world dataset, including:

- Loading the California Housing dataset from a public URL
- Exploratory data analysis (EDA) on the target and features
- Train/test splitting with a custom utility
- Training a depth-limited regression tree from scratch
- Evaluating performance with R² and MSE on both train and test sets
- Visualizing predicted vs. true values and residuals
- Discussing the bias–variance tradeoff in tree-based regression

## What Is a Regression Tree?

A Regression Tree is a supervised learning model that predicts a continuous outcome by recursively partitioning the feature space into regions and assigning each region a constant prediction.

At each internal node, the model applies a rule of the form:

- Is feature_j less than or equal to some threshold?

Each leaf node stores the mean of the training targets that fall into that region, and that mean is used as the prediction for any new observation routed to that leaf.

### Prediction Rule

For a leaf containing samples y_1, y_2, ..., y_k, the prediction is:

y_hat = (1 / k) * sum_i y_i

### Split Criterion

At each node, the tree selects the feature and threshold that minimize the Mean Squared Error of the resulting children:

MSE = (1 / n) * sum_i (y_i − y_hat)^2

The algorithm greedily chooses the split that yields the greatest reduction in MSE.

### Stopping Conditions

Splitting stops when:
- Maximum depth is reached
- The node contains too few samples (min_samples_leaf)
- No split reduces error

This produces a piecewise-constant approximation to the regression function.

## Dataset Description

The California Housing dataset contains aggregated housing statistics from the 1990 U.S. Census. Each row represents a geographic block group, described by numerical attributes related to population, income, location, and housing characteristics.

### Features

- median_income — Median income of the block group
- housing_median_age — Median age of the houses
- total_rooms — Total rooms in the block group
- total_bedrooms — Total bedrooms in the block group
- population — Block group population
- households — Number of households
- latitude — Geographic latitude
- longitude — Geographic longitude

### Target Variable

- median_house_value — Median house value (continuous, in dollars)

This dataset is well suited for regression trees because:
- The target is continuous
- Feature–target relationships are nonlinear
- Geographic features (latitude/longitude) interact in non-additive ways
- Interpretability is important

## Exploratory Data Analysis (EDA)

EDA in this notebook focuses on:
- Distribution of the target variable (histogram of median house values), which is right-skewed
- Feature scale comparison (boxplot of all numerical predictors)

These visualizations confirm that:
- Most houses have moderate prices, with a smaller number of very expensive ones
- Features differ substantially in scale

Regression trees can naturally handle skewed targets because predictions are conditional means within partitioned regions, and they are invariant to monotonic feature transformations.

## Preprocessing

Before training the model:
- Numerical features and the target are separated
- The data is split into training and test sets using a custom train_test_split (80/20 split, random_state=42)

No standardization is applied, because tree splits depend only on ordering comparisons (≤ thresholds) and are invariant to monotonic feature scaling.

## Training the Regression Tree

The RegressionTree is trained with:
- max_depth = 5
- min_samples_leaf = 20

These hyperparameters control the bias–variance tradeoff:
- max_depth limits how many recursive splits the tree can make
- min_samples_leaf prevents the tree from creating leaves that overfit small subsets of the data

## Model Evaluation

Performance is evaluated using two complementary regression metrics:

- R² Score — fraction of variance in the target explained by the model
- Mean Squared Error (MSE) — average squared deviation between predictions and ground truth

Both metrics are reported on the training set and the test set so that overfitting can be diagnosed directly. The close agreement between training and test performance under this specific split is interpreted with caution: regression trees are high-variance models, and results can vary substantially under different data splits, deeper trees, or stronger validation protocols (e.g., cross-validation or spatial splits).

## Diagnostic Visualizations

The notebook includes two standard regression diagnostic plots:

### Predicted vs. True Values

A scatter plot of predicted values against true target values, with the y = x diagonal overlaid. Points lying close to the diagonal indicate accurate predictions. In this notebook, most points cluster near the diagonal, with slight deviations at extreme target values.

### Residuals vs. Predicted Values

A scatter plot of residuals (y_true − y_pred) against predicted values. Residuals are centered around zero with no strong nonlinear pattern, but their spread increases at higher predicted values — a sign of mild heteroscedasticity that is common in housing-price data, where extreme high values are inherently harder to predict.

## Bias–Variance Tradeoff

Regression trees illustrate the bias–variance tradeoff directly through their depth parameter:

- Shallow trees → high bias, low variance (oversmoothed predictions)
- Deep trees → low bias, high variance (overfitting)

The max_depth and min_samples_leaf parameters together control this tradeoff. Regression trees are powerful and interpretable but can be unstable, which motivates ensemble methods such as Bagging and Random Forests covered elsewhere in the project.

## Purpose of This Notebook

This notebook is designed to:

- Demonstrate a from-scratch regression tree implementation
- Reinforce the connection between recursive partitioning and piecewise-constant prediction
- Show how MSE-based splits and depth control shape the learned function
- Provide standard regression diagnostics (predicted vs. true, residual plots)
- Serve as a baseline before more complex tree ensembles in this project

## Notes

The regression tree is implemented from scratch in the rice_ml package

No scikit-learn regression models are used

Visualizations are created using Matplotlib

Mathematical expressions are written in plain text and GitHub-compatible LaTeX

This notebook complements the classification-tree, ensemble, and linear-model examples in the repository