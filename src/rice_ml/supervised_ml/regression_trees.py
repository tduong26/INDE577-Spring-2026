"""
Regression Tree

This module provides a from-scratch implementation of a regression tree
using a recursive binary splitting strategy. The model divides the feature
space into smaller regions and predicts a numeric response in each terminal
node by taking the average of the training targets within that region.

Overview
--------
- Built for instructional clarity and step-by-step understanding
- Uses only NumPy for a lightweight implementation
- Follows a familiar fit / predict / score workflow
- Designed to show how tree-based regression works internally

Main Features
-------------
- Binary splits on numerical input features
- Top-down greedy tree construction
- Variance reduction as the split-selection criterion
- Leaf predictions based on the mean response value
- User-controlled stopping rules such as maximum depth and minimum samples

Implementation Details
----------------------
- Each node stores split information or a terminal prediction
- Candidate split points are checked directly from observed feature values
- The algorithm chooses the split that produces the largest reduction
  in within-node variability
- The tree grows recursively until a stopping condition is reached

Practical Notes
---------------
- This version is meant for learning, not large-scale production use
- Searching across many possible split points can become expensive
- No pruning or post-processing is included beyond the chosen stopping rules
- The algorithm is deterministic because no random split selection is used

This module is intended to give a clear and readable reference for how
regression trees partition predictor space and produce piecewise-constant
predictions for continuous outcomes.
"""


from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _validate_inputs(X, y=None):
    """
    Validate input arrays and ensure matching sample sizes.
    """
    X = np.asarray(X, dtype=float)

    if y is None:
        return X

    y = np.asarray(y, dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    return X, y


# ---------------------------------------------------------------------
# Internal Tree Node
# ---------------------------------------------------------------------

class _TreeNode:
    """
    Internal node for Regression Tree.
    """

    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_TreeNode"] = None,
        right: Optional["_TreeNode"] = None,
        value: Optional[float] = None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.feature_index is None


# ---------------------------------------------------------------------
# Regression Tree
# ---------------------------------------------------------------------

class RegressionTree:
    """
    Regression Tree for continuous targets.

    The tree recursively partitions the feature space to minimize
    Mean Squared Error (variance) within each leaf.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree.
    min_samples_split : int
        Minimum number of samples required to split a node.
    min_samples_leaf : int
        Minimum samples required in each leaf.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.n_features_ = None
        self.tree_ = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit regression tree to training data.
        """
        X, y = _validate_inputs(X, y)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")

        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """
        Predict continuous targets.
        """
        if self.tree_ is None:
            raise TypeError("Call fit before predict.")

        X = _validate_inputs(X)
        return np.array([self._predict_row(x, self.tree_) for x in X])

    def score(self, X, y):
        """
        R² score.
        """
        X, y = _validate_inputs(X, y)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        return 1.0 - ss_res / ss_tot

    # ------------------------------------------------------------------
    # Tree Construction
    # ------------------------------------------------------------------

    def _grow_tree(self, X, y, depth: int) -> _TreeNode:
        n_samples = X.shape[0]

        # Stopping conditions
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return _TreeNode(value=y.mean())

        feature, threshold, masks = self._best_split(X, y)

        if feature is None:
            return _TreeNode(value=y.mean())

        left = self._grow_tree(X[masks[0]], y[masks[0]], depth + 1)
        right = self._grow_tree(X[masks[1]], y[masks[1]], depth + 1)

        return _TreeNode(
            feature_index=feature,
            threshold=threshold,
            left=left,
            right=right,
            value=y.mean(),
        )

    # ------------------------------------------------------------------
    # Best Split (MSE)
    # ------------------------------------------------------------------

    def _best_split(
        self, X, y
    ) -> Tuple[Optional[int], Optional[float], Tuple[np.ndarray, np.ndarray]]:

        best_mse = np.inf
        best_feature = None
        best_threshold = None
        best_masks = None

        for feature in range(self.n_features_):
            values = np.unique(X[:, feature])

            for threshold in values:
                left = X[:, feature] <= threshold
                right = ~left

                if (
                    left.sum() < self.min_samples_leaf
                    or right.sum() < self.min_samples_leaf
                ):
                    continue

                mse = self._split_mse(y[left], y[right])

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_masks = (left, right)

        if best_feature is None:
            return None, None, (None, None)

        return best_feature, best_threshold, best_masks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_mse(self, y_left, y_right):
        n = len(y_left) + len(y_right)

        mse_left = np.var(y_left) * len(y_left)
        mse_right = np.var(y_right) * len(y_right)

        return (mse_left + mse_right) / n

    def _predict_row(self, x, node: _TreeNode):
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value