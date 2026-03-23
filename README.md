
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
.
├── src/
│   └── rice_ml/
│       ├── supervised_learning/
│       ├── unsupervised_learning/
│       ├── processing/
│       └── __init__.py
│
├── notebooks/
│   ├── supervised_learning/
│   └── unsupervised_learning/
│
├── examples/
│
├── tests/
│   ├── unit/
│   └── ...
│
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Repository Overview

### `src/rice_ml/`
Core Python package implementing machine learning algorithms from scratch, including both supervised and unsupervised learning methods.

### `notebooks/`
Jupyter notebooks that explain the usage, behavior, and interpretation of implemented algorithms on real and synthetic datasets.

### `examples/`
Additional example scripts and demonstrations for model training, evaluation, and visualization.

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
- Ensemble Methods
- Neural Networks / Multilayer Perceptron (MLP)

### Unsupervised Learning
- K-Means Clustering
- DBSCAN
- Principal Component Analysis (PCA)
- Community Detection

### Processing Utilities
- Feature standardization
- Basic preprocessing transformations
- Helper functions for model preparation and evaluation

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

The notebooks and examples in this repository are designed as teaching and learning resources. They typically include:

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

All tests are located in the `tests/` directory and can be run with:

```bash
pytest -q
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
from rice_ml.processing.preprocessing import standardize
```

## Project Goals

This repository was built to:

Deepen understanding of machine learning algorithms through from-scratch implementation
Practice professional Python package organization
Integrate code, testing, and documentation into one reproducible project
Provide educational examples for supervised and unsupervised learning
Emphasize assumptions, interpretation, and algorithmic behavior
Support coursework in IND 577
Purpose

This repository serves as:

A from-scratch machine learning library
A course project repository for IND 577
A reference implementation of classical machine learning methods
A learning tool for understanding machine learning algorithms beyond built-in libraries

## License

This project is intended for educational use as part of IND 577.
Refer to the repository for additional licensing details.

## Author
Thomas Duong
Rice University
IND 577