# Logistic Regression — Breast Cancer Wisconsin Dataset

This directory contains a Jupyter notebook demonstrating binary logistic regression implemented entirely from scratch using the rice_ml package. No scikit-learn classifiers are used.

The notebook applies logistic regression to the Breast Cancer Wisconsin (Diagnostic) dataset, a well-known benchmark in medical classification, and emphasizes interpretability, probability modeling, and proper evaluation in a clinical context.

## What This Notebook Covers

The notebook provides a complete supervised learning workflow:
- Loading the Breast Cancer Wisconsin dataset from sklearn.datasets
- Exploratory data analysis (EDA)
- Feature preprocessing and standardization
- Training logistic regression using gradient descent
- Evaluating performance with multiple metrics
- Visualizing the decision boundary in PCA-reduced space
- Interpreting results from both statistical and clinical perspectives

All model components are implemented from scratch using the rice_ml package. Scikit-learn is used only for data loading and PCA-based visualization.

## Dataset Overview

**Dataset:** Breast Cancer Wisconsin (Diagnostic)
**Source:** sklearn.datasets / UCI ML Repository
**Task:** Binary classification (malignant vs. benign tumor)

### Features

Each observation represents a tumor sample with 30 numerical features computed from digitized images of fine needle aspirate (FNA) biopsies. Features describe properties of cell nuclei — radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension — each reported as mean, standard error, and worst value.

Target:
- 0 → Malignant
- 1 → Benign

### Data Notes

All 30 features are numeric

The dataset contains 569 samples with no missing values

Feature scales differ substantially (e.g., radius ≈ 10s vs. area ≈ 1000s)

Features within the same family (radius, perimeter, area) are highly correlated

## Exploratory Data Analysis (EDA)

EDA in this notebook focuses on:
- Class distribution (bar chart and pie chart)
- Feature distributions per class (histograms of mean features)
- Correlation structure among mean features (correlation heatmap)
- Class separability for key features (boxplots of radius, perimeter, area, concavity, concave points)

These observations motivate feature standardization, which is critical for stable gradient-based optimization.

## Preprocessing

Before training the model:
- Features and labels are separated
- Data is split into training and test sets using a custom train_test_split implementation (80/20 split)
- Features are standardized to zero mean and unit variance using a custom standardize function
- The scaler is fit only on the training set to prevent data leakage

This ensures fair evaluation and numerical stability.

## Logistic Regression Model

The custom LogisticRegression class implements:
- Sigmoid activation for probabilistic outputs
- Binary cross-entropy (log-loss)
- Gradient descent optimization
- Probability predictions and hard classification

The model is trained with a learning rate of 0.1 over 1000 iterations. Logistic regression models the probability of malignancy directly, making it especially useful for risk assessment in medical settings.

## Evaluation Metrics

The notebook evaluates model performance using:
- Accuracy — overall classification correctness
- Confusion Matrix — insight into false positives and false negatives
- Precision, Recall, F1 Score — class-specific performance for the benign class
- Specificity — true negative rate, important for clinical reliability

Special emphasis is placed on minimizing false negatives, since missed malignant cases are the clinically most costly error.

## Visualization

The notebook includes:
- Training loss curve over gradient descent iterations
- Top 15 feature coefficients ranked by magnitude
- Confusion matrix heatmaps (counts and normalized)
- Predicted probability distribution by class
- 2-D decision boundary plot in PCA-reduced feature space

## Interpreting Performance

Accuracy on this dataset is high (~96–97%) because:
- The two classes are well-separated in feature space
- Features are clean and well-engineered
- Standardization stabilizes gradient descent

The most influential features include worst concave points, mean concave points, and worst perimeter — consistent with clinical understanding of tumor morphology.

## Purpose of This Notebook

This notebook is designed to:
- Demonstrate a correct from-scratch implementation of logistic regression
- Highlight the importance of preprocessing and evaluation choices
- Reinforce the probabilistic interpretation of classification models
- Serve as an interpretable baseline for more complex models
- Provide a clear, reproducible example for INDE 577 coursework

## Notes

All models are implemented using the rice_ml package

No scikit-learn classifiers are used (only load_breast_cancer and PCA for visualization)

Visualizations rely on Matplotlib and Seaborn

The notebook complements unit tests and ensemble examples elsewhere in the repository