"""
perceptron.py

A simple, from-scratch implementation of the binary Perceptron classifier.

This version is intended for educational purposes and supports:
- binary classification with labels in {0, 1} or {-1, 1}
- batch training through repeated sample updates
- prediction and accuracy scoring

Typical usage
-------------
model = Perceptron(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = model.score(X_test, y_test)
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """
    A simple binary Perceptron classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used in weight updates.
    n_iterations : int, default=1000
        Number of passes through the training data.
    random_state : int, default=42
        Seed for reproducibility.

    Attributes
    ----------
    weights : np.ndarray
        Weight vector of shape (n_features,).
    bias : float
        Intercept term.
    errors_ : list[int]
        Number of misclassifications in each training epoch.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors_ = []

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Convert labels to {-1, 1} format.

        Accepts labels in {0, 1} or {-1, 1}.
        """
        y = np.asarray(y)

        unique_labels = np.unique(y)
        if set(unique_labels) == {0, 1}:
            return np.where(y == 0, -1, 1)
        if set(unique_labels) == {-1, 1}:
            return y.astype(int)

        raise ValueError("Perceptron supports only binary labels in {0,1} or {-1,1}.")

    def _activation(self, X: np.ndarray) -> np.ndarray:
        """Compute the linear output."""
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels in {0, 1}.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels in {0, 1}.
        """
        X = np.asarray(X, dtype=float)
        linear_output = self._activation(X)
        return np.where(linear_output >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Fit the Perceptron model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Binary target labels in {0,1} or {-1,1}.

        Returns
        -------
        Perceptron
            Fitted model.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("X and y must not be empty.")

        y_binary = self._prepare_labels(y)

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        self.errors_ = []

        for _ in range(self.n_iterations):
            errors = 0

            for xi, target in zip(X, y_binary):
                predicted = 1 if (np.dot(xi, self.weights) + self.bias) >= 0 else -1
                update = self.learning_rate * (target - predicted)

                if update != 0:
                    self.weights += update * xi
                    self.bias += update
                    errors += 1

            self.errors_.append(errors)

            if errors == 0:
                break

        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True labels in {0,1} or {-1,1}.

        Returns
        -------
        float
            Accuracy score.
        """
        y = np.asarray(y)
        y_true = np.where(self._prepare_labels(y) == -1, 0, 1)
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)