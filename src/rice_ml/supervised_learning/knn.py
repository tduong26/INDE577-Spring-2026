""" k-Nearest Neighbors (kNN)

This module provides from-scratch implementations of k-Nearest Neighbors
for both classification and regression using only NumPy.

kNN is a simple, non-parametric learning method that makes predictions
based on the labels or values of the nearest training points in the
feature space.

Overview
--------
- Pure NumPy implementation with no machine learning dependencies
- Supports both classification and regression tasks
- Includes explicit distance calculations for transparency
- Follows a familiar fit/predict/predict_proba/score API style

Models Included
---------------
- KNNClassifier:
  Predicts class labels using either majority voting or
  distance-weighted voting among the nearest neighbors.

- KNNRegressor:
  Predicts continuous outcomes by averaging the target values
  of nearby observations, with optional distance weighting.

Main Features
-------------
- Euclidean and Manhattan distance metrics
- User-defined number of neighbors
- Uniform or distance-based weighting
- Brute-force neighbor search for clarity and simplicity
- No training phase beyond storing the data

Implementation Details
----------------------
- Distances are computed directly to make the algorithm easier to study
- Neighbor selection is based on partial sorting of computed distances
- Input checks help ensure valid shapes and consistent dimensions
- Best suited for instructional use and smaller datasets

Practical Notes
---------------
- Because it uses brute-force search, runtime increases as the dataset grows
- This version does not include KD-trees, ball trees, or approximate search
- Performance may decrease in very high-dimensional settings

The goal of this module is to make the mechanics of kNN easy to read,
understand, and modify for learning purposes.
"""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["KNNClassifier", "KNNRegressor"]


def _validate_X(X, name: str = "X") -> np.ndarray:
    """Convert input into a non-empty 2D NumPy array of floats."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return X


def _validate_y(y, name: str = "y") -> np.ndarray:
    """Convert input into a 1D NumPy array."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return y


def _compute_distances(X_query: np.ndarray, X_train: np.ndarray, metric: str) -> np.ndarray:
    """Compute pairwise distances between query points and training points."""
    if metric == "euclidean":
        sq_query = np.sum(X_query ** 2, axis=1)[:, None]
        sq_train = np.sum(X_train ** 2, axis=1)[None, :]
        distances = np.sqrt(np.maximum(sq_query + sq_train - 2 * X_query @ X_train.T, 0.0))
        return distances

    if metric == "manhattan":
        return np.sum(np.abs(X_query[:, None, :] - X_train[None, :, :]), axis=2)

    raise ValueError("metric must be 'euclidean' or 'manhattan'.")


def _compute_weights(distances: np.ndarray, weights: str, eps: float = 1e-12) -> np.ndarray:
    """Convert distances into neighbor weights."""
    if weights == "uniform":
        return np.ones_like(distances, dtype=float)

    zero_mask = distances <= eps

    if np.any(zero_mask, axis=1).any():
        return np.where(zero_mask, 1.0, 0.0)

    return 1.0 / np.maximum(distances, eps)


class _KNNBase:
    """Base class for shared kNN functionality."""

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")

        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.weights = weights

        self._X = None
        self._y = None
        self.n_features_in_ = None

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Store the training data."""
        X = _validate_X(X, "X")
        y = _validate_y(y, "y")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")

        self._X = X
        self._y = y
        self.n_features_in_ = X.shape[1]
        return self

    def kneighbors(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return distances and indices of the k nearest neighbors."""
        if self._X is None:
            raise RuntimeError("The model must be fitted before calling kneighbors.")

        X = _validate_X(X, "X")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as the training data.")

        distances = _compute_distances(X, self._X, self.metric)

        neighbor_idx = np.argpartition(distances, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        neighbor_distances = np.take_along_axis(distances, neighbor_idx, axis=1)

        order = np.argsort(neighbor_distances, axis=1)
        neighbor_idx = np.take_along_axis(neighbor_idx, order, axis=1)
        neighbor_distances = np.take_along_axis(neighbor_distances, order, axis=1)

        return neighbor_distances, neighbor_idx


class KNNClassifier(_KNNBase):
    """k-Nearest Neighbors classifier."""

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit the classifier by storing training data and class labels."""
        super().fit(X, y)
        self.classes_ = np.unique(self._y)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities for each input sample."""
        distances, indices = self.kneighbors(X)
        weights = _compute_weights(distances, self.weights)
        neighbor_labels = self._y[indices]

        class_positions = np.searchsorted(self.classes_, neighbor_labels)
        n_classes = len(self.classes_)

        probabilities = (np.eye(n_classes)[class_positions] * weights[..., None]).sum(axis=1)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        y = _validate_y(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class KNNRegressor(_KNNBase):
    """k-Nearest Neighbors regressor."""

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit the regressor by storing training data and numeric targets."""
        X = _validate_X(X, "X")
        y = _validate_y(y, "y").astype(float)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")

        self._X = X
        self._y = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous target values."""
        distances, indices = self.kneighbors(X)
        weights = _compute_weights(distances, self.weights)
        neighbor_values = self._y[indices]

        weight_sums = weights.sum(axis=1)
        predictions = (weights * neighbor_values).sum(axis=1) / np.maximum(weight_sums, 1e-12)

        return predictions

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R^2."""
        y = _validate_y(y, "y").astype(float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            raise ValueError("R^2 is undefined when y is constant.")

        return float(1.0 - ss_res / ss_tot)