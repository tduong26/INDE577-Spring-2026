# Gradient Descent for Linear Regression — California Housing Dataset

This directory contains a Jupyter notebook demonstrating gradient descent implemented entirely from scratch using the custom rice_ml package.

The notebook trains a linear regression model on the California Housing dataset by minimizing the mean squared error (MSE) loss with first-order optimization.

No external machine learning libraries (e.g., scikit-learn) are used.

## Notebook Overview

Notebook: gradient_descent_example.ipynb
Optimizer: Custom GradientDescentND
Model: Linear regression with bias term
Dataset: California Housing — 1990 U.S. Census

The notebook walks through the complete workflow:

- Loading the California Housing dataset from a public URL
- Exploratory data analysis (EDA)
- Feature and target standardization
- Adding a bias column to the design matrix
- Defining the MSE gradient as a closure
- Running gradient descent until convergence
- Plotting the loss curve
- Evaluating performance with R² on the test set
- Visualizing actual vs. predicted values

## What Is Gradient Descent?

Gradient descent is an iterative optimization algorithm that minimizes a differentiable loss function by repeatedly stepping in the direction of steepest descent.

### Loss Function (MSE)

For linear regression with predictions y_hat = X w, the loss is:

L(w) = (1 / n) * sum_i (y_i − x_i^T w)^2

### Gradient

The gradient of the MSE loss is:

grad L(w) = (2 / n) * X^T (X w − y)

### Update Rule

At each iteration, weights are updated as:

w_new = w_old − alpha * grad L(w_old)

where alpha is the learning rate. Training stops when the weight update falls below a tolerance threshold or after a maximum number of iterations.

### Why Standardization Matters

Gradient descent is sensitive to feature scale. When features have very different magnitudes, a single learning rate cannot work well across all of them.

Standardizing features to zero mean and unit variance:

X_std = (X − mean) / std

puts all features on the same scale and lets the optimizer converge smoothly.

## Dataset Description

The California Housing dataset contains housing statistics from the 1990 U.S. Census. Each row represents a geographic block group.

### Features

- MedInc — Median income
- HouseAge — Median house age
- AveRooms — Average rooms per household
- AveBedrms — Average bedrooms per household
- Population — Block group population
- AveOccup — Average household occupancy
- Latitude — Block group latitude
- Longitude — Block group longitude

### Target

- MedHouseVal — Median house value (continuous)

### Data Notes

All features are numeric

The categorical ocean_proximity column is dropped and rows with missing values are removed

The target is right-skewed and capped at 5.0

Feature scales differ substantially, motivating standardization

## Exploratory Data Analysis (EDA)

EDA in this notebook focuses on:

- Distribution of the target variable (histogram with KDE, plus boxplot)
- Feature scale comparison (boxplot of all 8 features)
- Per-feature variance comparison (bar chart)

These plots make it clear that features differ widely in scale, which directly motivates standardization before training.

## Preprocessing

Before training the model:

- Features and the target are separated
- Both X and y are standardized to zero mean and unit variance using a custom standardize function
- Data is split into training and test sets using a custom train_test_split (80/20 split, random_state=42)
- The training and test design matrices are augmented with a leading column of ones to absorb the bias term into the weight vector

## Defining the Gradient as a Closure

The MSE gradient is defined as a closure that captures the training data X and y:

grad_f(w) → gradient vector

Benefits of this design:

- The GradientDescentND optimizer is fully generic — it knows nothing about the dataset or the loss
- The same optimizer can be reused for any differentiable loss
- Cleanly separates the optimizer from the problem being solved

## Training Procedure

The optimizer is configured with:

- learning_rate (alpha) = 0.01
- tol = 1e-6
- max_iter = 3000

Weights are initialized at the zero vector. The optimizer records every weight vector visited during training, which is later used to reconstruct the loss curve.

## Loss Curve

Because GradientDescentND only tracks weights (not loss values), the loss curve is reconstructed afterward by evaluating MSE at each saved weight vector.

A smooth, monotonically decreasing loss curve confirms that:

- The learning rate is well-chosen (no divergence)
- Standardization resolved the scale issue
- The optimizer converges before hitting max_iter

## Evaluation

Performance is evaluated on the test set using the coefficient of determination:

R² = 1 − SS_res / SS_tot

Where:
- R² = 1 → perfect predictions
- R² = 0 → no better than predicting the mean
- R² < 0 → worse than predicting the mean

The actual-vs-predicted scatter plot shows:
- A positive linear trend
- Increasing spread at higher target values
- Compression near the upper cap at 5.0

## Limitations

The linear model has known limitations on this dataset:

- Cannot capture nonlinear effects (e.g., proximity to coast, neighborhood interactions)
- The target's upper cap at 5.0 violates standard regression assumptions
- Full-batch gradient descent is slower per iteration than mini-batch variants on large datasets

## Purpose of This Notebook

This notebook is designed to:

- Demonstrate a correct from-scratch implementation of gradient descent
- Show why feature standardization matters for first-order optimization
- Illustrate clean separation between the optimizer and the loss function via the closure pattern
- Provide clear loss-curve and residual diagnostics
- Serve as a foundational optimization example for other supervised learning notebooks in the project

## Notes

GradientDescentND is implemented from scratch in the rice_ml package

Preprocessing utilities (standardize, train_test_split) and the R² metric come from rice_ml

No scikit-learn or other external ML libraries are used

Visualizations rely on Matplotlib and Seaborn

This notebook complements the logistic regression, perceptron, MLP, and tree-based examples in the repository