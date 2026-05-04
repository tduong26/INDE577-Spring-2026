# Spring 2026: IND 577: Data Science and Machine Learning

### Name: Thomas Duong
### Course: IND 577

---

## Overview

This repository contains a **from-scratch machine learning library** developed for **IND 577**, together with a collection of **educational notebooks**, **examples**, and **tests** that demonstrate, evaluate, and validate core machine learning algorithms.

The project is designed to emphasize **algorithmic understanding, mathematical intuition, correctness, and clarity** rather than reliance on black-box implementations. All major methods are organized into a clean and modular Python package called **`rice_ml`**.

This repository serves both as a **learning-focused machine learning library** and as a **course codebase** for exploring classical supervised and unsupervised learning methods from first principles.

---

## Project Highlights

This repository includes:

- Fully custom implementations of core machine learning algorithms
- A modular and installable Python package: `rice_ml`
- Separate components for supervised learning, unsupervised learning, and preprocessing
- Educational notebooks and examples demonstrating each algorithm step by step
- A pytest-based test suite for validating correctness and expected behavior
- A structure designed for readability, reproducibility, and learning

---

## Repository Structure

```
INDE577-Spring-2026/
│
├── src/
│   └── rice_ml/                          # Core installable ML package
│       ├── __init__.py
│       ├── supervised_learning/
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── knn.py
│       │   ├── perceptron.py
│       │   ├── multilayer_perceptron.py
│       │   ├── decision_tree.py
│       │   ├── regression_trees.py
│       │   ├── ensemble_methods.py
│       │   ├── gradient_descent.py
│       │   └── distance_metrics.py
│       ├── unsupervised_learning/
│       │   ├── k_means_clustering.py
│       │   ├── dbscan.py
│       │   ├── pca.py
│       │   └── community_detection.py
│       └── processing/
│           ├── pre_processing.py         # Standardization, train/test split
│           └── post_processing.py        # Metrics (accuracy, R², MSE, etc.)
│
├── examples/                             # Jupyter notebooks per algorithm
│   ├── supervised_learning/
│   │   ├── Linear Regression/linear_regression_example.ipynb
│   │   ├── Logistic Regression/logistic_regression_example.ipynb
│   │   ├── KNN/knn_example.ipynb
│   │   ├── Perceptron/perceptron_example.ipynb
│   │   ├── Multilayer Perceptron/multilayer_perceptron_example.ipynb
│   │   ├── Decision Tree/decision_tree_example.ipynb
│   │   ├── Regression Trees/regression_trees_example.ipynb
│   │   ├── Ensemble Methods/ensemble_methods_example.ipynb
│   │   └── Gradient Descent/gradient_descent_example.ipynb
│   └── unsupervised_learning/
│       ├── K-Means Clustering/k_means_clustering_example.ipynb
│       ├── DBScan/dbscan_example.ipynb
│       ├── PCA/pca_example.ipynb
│       └── Community Detection/community_detection_example.ipynb
│
├── tests/                                # pytest test suite
│
├── .github/
│   └── workflows/
│       └── pr-tests.yml                  # CI runs pytest on push and PR
│
├── pyproject.toml                        # Build config and dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## Repository Overview

### `src/rice_ml/`
Core Python package implementing machine learning algorithms from scratch, including both supervised and unsupervised learning methods.

### `examples/`
Jupyter notebooks demonstrating model training, evaluation, and visualization for each implemented algorithm.

### `tests/`
Unit tests written with `pytest` to verify correctness, numerical behavior, and edge cases of the implemented methods.

### `pyproject.toml`
Project configuration and dependency management.

---

## Implemented Methods

The `rice_ml` package includes implementations of important machine learning algorithms such as:

### Supervised Learning
- Linear Regression
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Perceptron
- Decision Trees
- Regression Trees
- Ensemble Methods (Bagging, Voting, Random Forest)
- Gradient Descent (1D and N-D)
- Multilayer Perceptron (MLP)

### Unsupervised Learning
- K-Means Clustering
- DBSCAN
- Principal Component Analysis (PCA)
- Community Detection (Label Propagation)

### Processing Utilities
- Feature standardization and min-max scaling
- Train/test splitting
- Evaluation metrics (accuracy, R², MSE, confusion matrix)

---

## Design Philosophy

This project prioritizes:

- Algorithmic transparency over black-box abstraction
- Readable and well-structured code suitable for learning and inspection
- Separation of concerns between implementation, experimentation, and testing
- Reproducibility through explicit configuration and organized workflows
- Interpretability and clarity over unnecessary complexity

The goal is not only to build working machine learning models, but also to understand how they work internally.

---

## Notebooks and Examples

The notebooks in `examples/` are designed as teaching and learning resources. They typically include:

- Dataset loading and exploration
- Preprocessing and scaling
- Training custom models from `rice_ml`
- Visualizing predictions, clusters, or low-dimensional embeddings
- Interpreting results and discussing limitations

These materials are intended to support both conceptual understanding and practical implementation.

---

## Installation

From the repository root, install the package in editable mode with:

```bash
pip install -e .
```

## Running Tests

From the repository root, install the test dependencies and run pytest:

```bash
pip install -e ".[test]"
pytest
```

The test suite is intended to check:

- Numerical correctness
- Input validation
- Output consistency
- Edge cases
- Expected behavior on small datasets

## Example Usage

```python
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.supervised_learning.logistic_regression import LogisticRegression
from rice_ml.unsupervised_learning.k_means_clustering import KMeans
from rice_ml.processing.pre_processing import standardize
```

## Project Goals

This repository was built to:

- Deepen understanding of machine learning algorithms through from-scratch implementation
- Practice professional Python package organization
- Integrate code, testing, and documentation into one reproducible project
- Provide educational examples for supervised and unsupervised learning
- Emphasize assumptions, interpretation, and algorithmic behavior
- Support coursework in IND 577

## Purpose

This repository serves as:

- A from-scratch machine learning library
- A course project repository for IND 577
- A reference implementation of classical machine learning methods
- A learning tool for understanding machine learning algorithms beyond built-in libraries

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Thomas Duong**
Rice University
IND 577
