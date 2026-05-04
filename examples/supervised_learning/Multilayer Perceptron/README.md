# Multilayer Perceptron (MLP) — Ionosphere Dataset

This directory contains a Jupyter notebook demonstrating a Multilayer Perceptron (MLP) classifier implemented entirely from scratch using the rice_ml package.

The notebook builds directly on the single-layer Perceptron and shows how adding a hidden layer and nonlinear activation functions allows the model to learn nonlinear decision boundaries.

No scikit-learn neural networks are used.

## What This Notebook Covers

This notebook presents a complete supervised learning workflow:
1. Reviewing the limitations of the single-layer Perceptron
2. Introducing hidden layers and nonlinear activations
3. Training an MLP classifier from scratch using gradient descent and backpropagation
4. Comparing performance against a Perceptron baseline on the same train/test split
5. Visualizing test-set errors using PCA
6. Interpreting results from a geometric and learning-theoretic perspective

All code relies exclusively on custom implementations in the rice_ml package.

## Why Go Beyond the Perceptron?

The single-layer Perceptron can only learn linearly separable decision boundaries.

As a result, it struggles when:
- The classes are not linearly separable
- The relationship between predictors and the outcome is nonlinear
- Useful patterns appear only through interactions among features

The Multilayer Perceptron overcomes these limitations by introducing:
- One or more hidden layers
- Nonlinear activation functions
- A differentiable loss function optimized with backpropagation

## Model Overview: Multilayer Perceptron

An MLP consists of:
- An input layer
- One or more hidden layers
- An output layer

Each layer applies a linear transformation followed by a nonlinear activation.

### Forward Pass (Conceptual)

For a network with one hidden layer:

Z1 = X · W1 + b1
A1 = phi(Z1)
Z2 = A1 · W2 + b2
y_hat = sigmoid(Z2)

Where:

phi is a nonlinear hidden-layer activation function (e.g., ReLU)

sigmoid maps the output to a probability for binary classification

### Activation Functions

Activation functions introduce nonlinearity into the model.

ReLU (hidden layers)
phi(z) = max(0, z)

Sigmoid (output layer)
sigmoid(z) = 1 / (1 + exp(-z))

Without nonlinear activations, stacking layers would collapse into a single linear model.

### Loss Function

For binary classification, the MLP minimizes binary cross-entropy loss:

loss = -(1/n) sum_i [ y_i · log(y_hat_i) + (1 − y_i) · log(1 − y_hat_i) ]

This loss:

Penalizes confident incorrect predictions heavily

Produces smooth gradients for optimization

### Training with Gradient Descent and Backpropagation

Model parameters are updated using gradient descent:

w = w − learning_rate · gradient

Gradients are computed efficiently using backpropagation, which applies the chain rule to propagate errors backward through the network. Backpropagation makes training hidden-layer models computationally feasible.

## Dataset Overview

The notebook applies both the Perceptron and the MLP to the Ionosphere dataset from the UCI Machine Learning Repository.

### Dataset Characteristics

Number of samples: 351
Number of features: 34 continuous numerical values
Target labels:
- 1 → good radar return
- 0 → bad radar return

Moderate class balance

No missing values

This dataset is not perfectly linearly separable, making it ideal for demonstrating the advantages of nonlinear models.

## Exploratory Data Analysis (EDA)

EDA focuses on:
- Target class distribution (bar chart of class counts)
- Feature scale differences across the first 10 predictors (boxplot)

Neural networks are sensitive to feature scale, so understanding these properties is essential before training.

## Preprocessing

All features are standardized to zero mean and unit variance:

X_std = (X − mean) / std

This ensures:
- Equal contribution from all features
- Stable gradient updates
- Faster convergence during training

The dataset is then split into training and test sets using a custom train_test_split function (80/20 split, random_state=42).

## Baseline Model: Perceptron

Before training the MLP, a single-layer Perceptron is trained as a baseline with:
- learning_rate = 0.1
- max_iter = 1000
- random_state = 42

This comparison isolates the effect of:
- Hidden layers
- Nonlinear activations
- Backpropagation

The Perceptron establishes a lower bound on performance.

## Training the Multilayer Perceptron

The MLP is trained with:
- One hidden layer of 16 units
- learning_rate = 0.01
- max_iter = 3000
- random_state = 42
- Sigmoid output activation for binary classification
- Gradient descent with backpropagation

The model learns nonlinear feature interactions that the Perceptron cannot represent.

## Model Evaluation

Performance is evaluated using classification accuracy:

Accuracy = (number of correct predictions) / (total samples)

The Perceptron and MLP are evaluated on the same test set, and the results are displayed side-by-side in a bar chart for direct comparison. The MLP consistently outperforms the Perceptron, demonstrating the benefit of nonlinear modeling.

## PCA Visualization of Test-Set Predictions

Because the dataset has 34 features, direct visualization is not possible.

Principal Component Analysis (PCA) — also implemented in rice_ml — is used to project the standardized test data into two dimensions. Correctly classified and misclassified points are then plotted separately to show where the MLP's errors concentrate in feature space.

Important notes:
- PCA does not affect model training
- PCA is fit on the training set and applied to the test set
- The MLP is trained in the full 34-dimensional feature space
- Overlap in PCA space helps explain why accuracy is not perfect

## Limitations

Despite improved performance, the MLP still has limitations:
- Sensitive to hyperparameter choices (hidden layer size, learning rate, iterations)
- Requires careful feature scaling
- Can overfit without regularization
- Less interpretable than linear models

These issues motivate deeper architectures and regularization techniques in modern deep learning.

## Purpose of This Notebook

This notebook is designed to:
- Demonstrate a correct from-scratch MLP implementation
- Highlight the limitations of linear classifiers
- Show how hidden layers enable nonlinear learning
- Reinforce the role of backpropagation
- Serve as a conceptual bridge to modern neural networks

## Notes

All models are implemented using the rice_ml package

PCA for visualization is also from the rice_ml package

No scikit-learn neural networks are used

Visualizations rely on Matplotlib

This notebook complements the Perceptron and Logistic Regression examples in the repository