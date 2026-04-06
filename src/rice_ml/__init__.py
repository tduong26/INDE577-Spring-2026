"""
rice_ml

A from-scratch machine learning library created for learning,
experimentation, and clear understanding of core algorithms.

The package includes implementations for supervised learning,
unsupervised learning, and preprocessing tools, all organized
with a simple, sklearn-style interface.
"""

# Subpackages
from . import supervised_learning
from . import unsupervised_learning
from . import processing

# Supervised learning models
from .supervised_learning import (
    LinearRegression,
    LogisticRegression,
    KNNClassifier,
    KNNRegressor,
    Perceptron,
    MultilayerPerceptron,
    DecisionTreeClassifier,
    DecisionTree,
    RegressionTree,
)

# Unsupervised learning models
from .unsupervised_learning import (
    KMeans,
    DBSCAN,
    PCA,
    LabelPropagation,
)

# Processing and utility functions
from .processing import (
    standardize,
    minmax_scale,
    train_test_split,
    accuracy_score,
    r2_score,
    mean_squared_error,
    confusion_matrix,
)

__all__ = [
    # Subpackages
    "supervised_learning",
    "unsupervised_learning",
    "processing",

    # Supervised learning models
    "LinearRegression",
    "LogisticRegression",
    "KNNClassifier",
    "KNNRegressor",
    "Perceptron",
    "MultilayerPerceptron",
    "DecisionTreeClassifier",
    "DecisionTree",
    "RegressionTree",

    # Unsupervised learning models
    "KMeans",
    "DBSCAN",
    "PCA",
    "LabelPropagation",

    # Processing and utility functions
    "standardize",
    "minmax_scale",
    "train_test_split",
    "accuracy_score",
    "r2_score",
    "mean_squared_error",
    "confusion_matrix",
]