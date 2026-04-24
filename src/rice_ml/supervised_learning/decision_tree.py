"""decision_tree.py

A from-scratch Decision Tree Classifier for educational use.

Features
--------
- Binary splits on numeric features
- Gini impurity or entropy criterion
- Maximum depth / minimum samples stopping rules
- Class prediction and probability prediction
- Accuracy scoring
- Simple text-tree display

This implementation is intended for small to medium teaching examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _TreeNode:
    """Single node in the decision tree."""

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    value: Optional[int] = None
    proba: Optional[np.ndarray] = None
    n_samples: int = 0
    impurity: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """A simple decision tree classifier.

    Parameters
    ----------
    criterion : str, default="gini"
        Splitting criterion. Options are "gini" and "entropy".
    max_depth : int or None, default=None
        Maximum tree depth. If None, expand until other stopping rules apply.
    min_samples_split : int, default=2
        Minimum number of samples needed to attempt a split.
    min_samples_leaf : int, default=1
        Minimum number of samples allowed in each leaf.
    max_features : int or None, default=None
        Number of features to consider at each split. If None, use all features.
    random_state : int or None, default=None
        Random seed for feature subsampling.

    Attributes
    ----------
    tree_ : _TreeNode
        Root node of the fitted tree.
    n_classes_ : int
        Number of classes in the target.
    n_features_in_ : int
        Number of features in the training data.
    feature_importances_ : ndarray
        Normalized impurity-based feature importance scores.
    classes_ : ndarray
        Original class labels.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.tree_: Optional[_TreeNode] = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_X(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return X

    def _validate_y(self, y) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        return y

    # ------------------------------------------------------------------
    # Impurity functions
    # ------------------------------------------------------------------
    def _class_counts(self, y: np.ndarray) -> np.ndarray:
        return np.bincount(y, minlength=self.n_classes_)

    def _impurity(self, y: np.ndarray) -> float:
        counts = self._class_counts(y)
        probs = counts / len(y)
        probs = probs[probs > 0]

        if self.criterion == "gini":
            return 1.0 - np.sum(probs**2)
        return -np.sum(probs * np.log2(probs))

    def _leaf_value(self, y: np.ndarray) -> tuple[int, np.ndarray]:
        counts = self._class_counts(y)
        proba = counts / counts.sum()
        value = int(np.argmax(counts))
        return value, proba

    # ------------------------------------------------------------------
    # Split search
    # ------------------------------------------------------------------
    def _candidate_features(self) -> np.ndarray:
        n_features = self.n_features_in_
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=self.max_features, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[Optional[int], Optional[float], float]:
        parent_impurity = self._impurity(y)
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_samples = X.shape[0]
        candidate_features = self._candidate_features()

        for feature_index in candidate_features:
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            if unique_values.size <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                left_impurity = self._impurity(y_left)
                right_impurity = self._impurity(y_right)
                child_impurity = (
                    (n_left / n_samples) * left_impurity
                    + (n_right / n_samples) * right_impurity
                )
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        impurity = self._impurity(y)
        leaf_value, leaf_proba = self._leaf_value(y)
        node = _TreeNode(
            value=leaf_value,
            proba=leaf_proba,
            n_samples=len(y),
            impurity=impurity,
        )

        if len(np.unique(y)) == 1:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split:
            return node

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain <= 0:
            return node

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.value = None
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        self.feature_importances_[best_feature] += best_gain * len(y)
        return node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit the decision tree classifier."""
        X = self._validate_X(X)
        y = self._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        self.tree_ = self._build_tree(X, y_encoded, depth=0)

        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            self.feature_importances_ /= total_importance

        return self

    def _predict_one(self, x: np.ndarray, node: _TreeNode) -> tuple[int, np.ndarray]:
        if node.is_leaf:
            return node.value, node.proba

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X) -> np.ndarray:
        """Predict class labels for X."""
        if self.tree_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = self._validate_X(X)
        preds = [self._predict_one(x, self.tree_)[0] for x in X]
        return self.classes_[np.array(preds, dtype=int)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X."""
        if self.tree_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = self._validate_X(X)
        probas = [self._predict_one(x, self.tree_)[1] for x in X]
        return np.vstack(probas)

    def score(self, X, y) -> float:
        """Return classification accuracy."""
        y = self._validate_y(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------
    def print_tree(self, feature_names: Optional[list[str]] = None, decimals: int = 3) -> None:
        """Print a simple text representation of the fitted tree."""
        if self.tree_ is None:
            raise ValueError("Model has not been fitted yet.")

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        def _recurse(node: _TreeNode, depth: int) -> None:
            indent = "  " * depth
            if node.is_leaf:
                pred_label = self.classes_[node.value]
                print(
                    f"{indent}Leaf(samples={node.n_samples}, "
                    f"predict={pred_label}, impurity={node.impurity:.{decimals}f})"
                )
                return

            name = feature_names[node.feature_index]
            print(
                f"{indent}if {name} <= {node.threshold:.{decimals}f} "
                f"(samples={node.n_samples}, impurity={node.impurity:.{decimals}f}):"
            )
            _recurse(node.left, depth + 1)
            print(f"{indent}else:")
            _recurse(node.right, depth + 1)

        _recurse(self.tree_, 0)

# After the DecisionTreeClassifier class definition
DecisionTree = DecisionTreeClassifier