"""
multilayer_perceptron.py

A simple, from-scratch implementation of a one-hidden-layer
Multilayer Perceptron (MLP) for binary classification.

This version supports:
- one hidden layer
- sigmoid activation in the hidden layer
- sigmoid output for binary classification
- batch gradient descent
- probability prediction, class prediction, and accuracy scoring

Typical usage
-------------
model = MultilayerPerceptron(
    n_hidden=16,
    learning_rate=0.01,
    n_iterations=5000
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
preds = model.predict(X_test)
acc = model.score(X_test, y_test)
"""

from __future__ import annotations

import numpy as np


class MultilayerPerceptron:
    """
    A one-hidden-layer neural network for binary classification.

    Parameters
    ----------
    n_hidden : int, default=16
        Number of neurons in the hidden layer.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=5000
        Number of training iterations.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    W1 : np.ndarray
        Weights from input layer to hidden layer.
    b1 : np.ndarray
        Bias for hidden layer.
    W2 : np.ndarray
        Weights from hidden layer to output layer.
    b2 : float
        Bias for output layer.
    losses_ : list[float]
        Training loss history.
    """

    def __init__(
        self,
        n_hidden: int = 16,
        learning_rate: float = 0.01,
        n_iterations: int = 5000,
        random_state: int = 42,
    ):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.losses_ = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation with clipping for numerical stability."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid using activated values."""
        return a * (1.0 - a)

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize weights and biases."""
        rng = np.random.default_rng(self.random_state)

        self.W1 = rng.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = rng.normal(loc=0.0, scale=0.1, size=(self.n_hidden, 1))
        self.b2 = np.zeros((1, 1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultilayerPerceptron":
        """
        Fit the MLP using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Binary labels in {0, 1}.

        Returns
        -------
        MultilayerPerceptron
            Fitted model.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("X and y must not be empty.")

        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("MLP currently supports binary labels in {0, 1} only.")

        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        self._initialize_parameters(n_features)
        self.losses_ = []

        for i in range(self.n_iterations):
            # Forward pass
            z1 = X @ self.W1 + self.b1
            a1 = self._sigmoid(z1)

            z2 = a1 @ self.W2 + self.b2
            a2 = self._sigmoid(z2)

            # Loss
            loss = self._binary_cross_entropy(y, a2)
            self.losses_.append(loss)

            # Backward pass
            dz2 = a2 - y
            dW2 = (a1.T @ dz2) / n_samples
            db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples

            dz1 = (dz2 @ self.W2.T) * self._sigmoid_derivative(a1)
            dW1 = (X.T @ dz1) / n_samples
            db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples

            # Gradient update
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}, Loss: {loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for the positive class.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)

        z1 = X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)

        return a2.ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels in {0, 1}.
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True binary labels.

        Returns
        -------
        float
            Accuracy score.
        """
        y = np.asarray(y).astype(int)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)