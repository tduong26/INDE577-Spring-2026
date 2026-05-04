# Ensemble Methods — Ionosphere Dataset

This directory contains a Jupyter notebook demonstrating three ensemble learning strategies implemented entirely from scratch using the custom rice_ml package.

The notebook applies Bagging, Random Forest, and Voting classifiers to the Ionosphere dataset, a real-world radar signal classification problem, and benchmarks each ensemble against a standalone Decision Tree baseline.

No scikit-learn classifiers are used.

## Notebook Overview

Notebook: ensemble_methods_example.ipynb
Models: BaggingClassifier, RandomForestClassifier, VotingClassifier (all custom)
Baseline: DecisionTreeClassifier (custom)
Dataset: Ionosphere — UCI Machine Learning Repository

The notebook walks through the complete process of applying ensemble methods to a real-world dataset, including:

- Loading the Ionosphere dataset directly from the web
- Exploratory data analysis (EDA)
- Feature preprocessing and standardization
- Training a Decision Tree baseline
- Training three ensemble models (Bagging, Random Forest, Voting)
- Comparing all models on classification accuracy
- Visualizing the data using PCA for intuition

## What Are Ensemble Methods?

Ensemble methods combine the outputs of several individual models into a single, more powerful predictor. The notebook focuses on three complementary strategies:

### Bagging (Bootstrap Aggregating)

Bagging trains multiple base learners (here, decision trees) on bootstrap samples of the training data and aggregates their predictions by majority vote. This reduces variance without increasing bias, making it ideal for high-variance learners like decision trees.

### Random Forest

Random Forest extends Bagging by decorrelating trees through randomized feature selection at each split. By forcing trees to consider different subsets of features, the ensemble produces less correlated errors and benefits more strongly from variance reduction.

### Voting Classifier

The Voting Classifier combines heterogeneous models by majority vote. Unlike Bagging and Random Forest, which average many models of the same type, Voting pools predictions from qualitatively different learners — here, a Decision Tree, a Logistic Regression, and a Random Forest. Using an odd number of voters avoids ties.

### Why Ensembles Work

For an ensemble of T approximately independent base models each with variance σ², the variance of the averaged predictor is:

Var(h_bar) ≈ σ² / T

This shows that adding more (sufficiently independent) models directly reduces variance. Combined with the fact that individual model errors tend to cancel out when aggregated, ensembles consistently improve generalization over single high-variance learners.

The majority voting rule for hard-label ensembles is:

y_hat_ensemble = argmax_k sum_t 1[y_hat_t = k]

## Dataset Description

The Ionosphere dataset contains radar signal returns collected by a phased array antenna. Each observation represents a radar pulse described by 34 continuous features derived from signal processing.

### Features

- 34 continuous numerical attributes from radar signal processing

### Target Variable

- 1 → good radar return
- 0 → bad radar return

### Data Notes

The feature space is moderately high-dimensional

Decision boundaries are nonlinear

Individual decision trees tend to overfit on this dataset

The dataset is moderately imbalanced toward "good" returns

This combination makes Ionosphere a strong testbed for variance-reducing ensemble methods.

## Exploratory Data Analysis (EDA)

EDA in this notebook focuses on:

- Dataset structure and missing-value check
- Class distribution (bar chart of good vs. bad returns)
- Feature scale comparison (boxplot of all 34 features before standardization)

The boxplot reveals that feature ranges vary significantly across dimensions, with some features tightly concentrated near zero and others spanning much wider numeric ranges. This motivates standardization before training, especially for any distance-based component (e.g., the KNN-style behavior present in some ensemble compositions).

## Preprocessing

Before training the models:

- Features and labels are separated
- Features are standardized to zero mean and unit variance using a custom standardize function
- Data is split into training and test sets using a custom train_test_split (80/20 split, random_state=42)

Although decision trees are scale-invariant, standardizing the data provides consistency across heterogeneous models in the Voting ensemble and improves numerical stability.

## Baseline Model — Decision Tree

A single Decision Tree (max_depth=None, fully grown) is trained as the baseline.

Decision trees are:
- Flexible and nonlinear
- Easy to interpret
- High variance (unstable across data resamples)

The high-variance property is precisely what makes them ideal base learners for variance-reducing ensembles.

## Ensemble Models

The notebook trains three ensemble models with the following configurations:

### Bagging Classifier
- Base learner: DecisionTreeClassifier
- n_estimators = 15
- random_state = 0

### Random Forest Classifier
- n_estimators = 15
- random_state = 0

### Voting Classifier
Combines three heterogeneous learners:
- DecisionTreeClassifier (max_depth=5)
- LogisticRegression (max_iter=5000, learning_rate=0.1)
- RandomForestClassifier (n_estimators=15, random_state=0)

## Model Evaluation

All four models — Decision Tree baseline plus the three ensembles — are evaluated on the same test set using classification accuracy:

Accuracy = (number of correct predictions) / (total samples)

Results are presented in a side-by-side comparison table and a bar chart for direct visual comparison.

### Typical Findings

- Bagging and Random Forest clearly outperform the standalone Decision Tree, confirming that variance reduction translates into measurable accuracy gains
- Random Forest typically edges out Bagging due to additional decorrelation from feature subsampling
- The Voting Classifier improves over the single Decision Tree but tends to fall slightly short of Bagging and Random Forest because:
  - It aggregates only a handful of models rather than many bootstrapped trees, limiting its variance-reduction capacity
  - Hard majority voting can override a strong model's correct prediction when the remaining voters disagree

These outcomes are consistent with the bias–variance tradeoff: decision trees have low bias but high variance, so ensemble strategies that reduce variance without inflating bias produce the largest gains.

## PCA Visualization

Because the dataset has 34 features, direct visualization is not possible.

The notebook applies Principal Component Analysis (PCA) — also implemented in rice_ml — to project the standardized data onto its first two principal components. Points are colored by class to provide intuition about class overlap.

Important notes:
- PCA is used only for visualization
- All models are trained in the full 34-dimensional feature space
- Visible overlap in the PCA projection helps explain why no model achieves perfect accuracy

## Purpose of This Notebook

This notebook is designed to:

- Demonstrate correct from-scratch implementations of Bagging, Random Forest, and Voting
- Make the variance-reduction mechanism behind ensembles concrete and measurable
- Show how heterogeneous and homogeneous ensembles differ in practice
- Provide a clear empirical demonstration of the bias–variance tradeoff
- Serve as a capstone example that builds on the Decision Tree, Logistic Regression, and KNN notebooks elsewhere in the repository

## Notes

All ensemble models are implemented from scratch in the rice_ml package

Base learners (DecisionTreeClassifier, LogisticRegression, KNNClassifier) and PCA are also from rice_ml

Preprocessing utilities (standardize, train_test_split) and the accuracy_score metric come from rice_ml

No scikit-learn classifiers are used

Visualizations rely on Matplotlib

This notebook complements the Decision Tree, Logistic Regression, KNN, and Perceptron examples in the repository