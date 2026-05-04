# Perceptron Classifier 

This directory contains a Jupyter notebook demonstrating the Perceptron algorithm implemented entirely from scratch using the rice_ml package. No scikit-learn classifiers are used to train the Perceptron.

The notebook applies the Perceptron to the Ionosphere dataset, a real-world radar signal classification task, and focuses on linear decision boundaries, mistake-driven learning, and quantitative comparison against other linear classifiers.

## What This Notebook Covers

The notebook presents a complete supervised learning workflow:
- Loading a real dataset from a public URL
- Exploratory data analysis (EDA)
- Feature preprocessing and standardization
- Training a Perceptron classifier from scratch
- Evaluating classification accuracy
- Visualizing high-dimensional data using PCA
- Comparing the Perceptron quantitatively against Logistic Regression and Linear SVM
- Interpreting results through theory and geometry

The Perceptron and core utilities come from the custom rice_ml package. Scikit-learn is used only as a baseline reference in the comparison section.

## Model Overview: The Perceptron

The Perceptron is one of the earliest learning algorithms in machine learning, introduced by Frank Rosenblatt in 1958. It learns a linear decision boundary of the form:

f(x) = w · x + b

Predictions are made as:

f(x) >= 0 → class 1

f(x) < 0 → class 0

### Learning Rule

For each training example:

If the prediction is correct → no update

If the prediction is incorrect → update weights and bias

w = w + η (y − ŷ) x
b = b + η (y − ŷ)

where:

η is the learning rate
y is the true label
ŷ is the predicted label

### Key Properties

Updates occur only on misclassified samples

No explicit loss function is minimized

Guaranteed to converge only if the data are linearly separable

## Dataset Overview

Dataset: Ionosphere
Source: UCI Machine Learning Repository
Task: Binary classification

Characteristics
- Samples: 351
- Features: 34 continuous numerical values

Target:
- 1 → good radar return
- 0 → bad radar return

No missing values

Moderate class balance

This dataset is useful for studying linear classifiers, but the classes are not perfectly separable, which limits Perceptron performance.

## Exploratory Data Analysis (EDA)

EDA in this notebook focuses on:
- Target class balance (bar chart of good vs. bad returns)
- Feature scale differences (boxplot of all 34 features before standardization)
- Identifying the need for preprocessing

Because Perceptron updates are proportional to feature values, differences in feature scale can strongly influence learning behavior.

## Preprocessing

Before training:
- Features are standardized to zero mean and unit variance
- The dataset is split into training and test sets using a custom train_test_split function (80/20 split)

This ensures:
- Balanced feature contributions
- Stable updates
- Faster convergence

## Model Training

The Perceptron is trained with:
- learning_rate = 0.1
- max_iter = 1000
- random_state = 42 (for reproducibility)

The model iteratively refines a linear decision boundary based on misclassifications in the training set.

## Model Evaluation

Model performance is evaluated using classification accuracy:

Accuracy = (number of correct predictions) / (total samples)

Accuracy is appropriate here due to the moderately balanced classes.

## Visualization with PCA

Because the dataset has 34 features, direct visualization is not possible.

The notebook applies Principal Component Analysis (PCA) — also implemented in rice_ml — to project the data into two dimensions for visualization only.

Important notes:
- PCA is used only for visualization
- The Perceptron is trained in the full 34-dimensional feature space
- Overlap in the PCA projection explains why accuracy is limited

## Quantitative Comparison with Other Linear Classifiers

To put the Perceptron's performance in context, the notebook trains three linear classifiers on the same train/test split and compares them on accuracy, training time, and confusion matrices:

| Model | Learning Strategy | Output | Loss Function |
|---|---|---|---|
| Perceptron (rice_ml) | Mistake-driven updates | Hard label | None (implicit) |
| Logistic Regression | Gradient descent on log-loss | Probability | Cross-entropy |
| Linear SVM | Margin maximization | Hard label | Hinge loss |

Findings:
- The Perceptron is the fastest to train but yields the lowest accuracy
- Logistic Regression produces a more stable boundary by optimizing a smooth global objective
- Linear SVM typically achieves the best accuracy by maximizing the margin between classes
- All three models tend to misclassify more "Bad" samples than "Good" ones, reflecting mild class imbalance and within-class variance

## Limitations of the Perceptron

While simple and interpretable, the Perceptron has important limitations:
- Learns only linear decision boundaries
- Cannot solve non-linearly separable problems
- Produces no probabilistic outputs
- Sensitive to noisy or overlapping data

These limitations motivate more advanced models such as:
- Logistic Regression
- Support Vector Machines
- Neural Networks

## Purpose of This Notebook

This notebook is designed to:
- Demonstrate a correct from-scratch implementation of the Perceptron
- Reinforce the concept of linear separability
- Provide intuition for gradient-based learning
- Quantify the trade-offs between simple and more sophisticated linear classifiers
- Serve as a foundational supervised learning example
- Support comparison with more advanced classifiers in the project

## Notes

The Perceptron is implemented using the rice_ml package

PCA for visualization is also from the rice_ml package

Scikit-learn is used only as a comparison baseline (Logistic Regression, Linear SVM)

Visualizations rely on Matplotlib

This notebook complements logistic regression and ensemble examples in the repository