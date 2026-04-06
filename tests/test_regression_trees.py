import numpy as np
import pytest

from rice_ml.supervised_ml.regression_tree import RegressionTree


# ---------------------------------------------------------------------
# Basic API Tests
# ---------------------------------------------------------------------
def test_fit_and_predict_shapes():
    X = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y = 2 * X.ravel() + 1

    model = RegressionTree(max_depth=3)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_predict_before_fit_raises():
    model = RegressionTree()
    X = np.array([[0.0]])

    with pytest.raises(ValueError):
        model.predict(X)


# ---------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------
def test_perfect_fit_simple_linear():
    X = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y = 2 * X.ravel() + 1

    model = RegressionTree(max_depth=5)
    model.fit(X, y)

    r2 = model.score(X, y)
    assert r2 > 0.95


def test_nonlinear_data_fit():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([1, 1, 2, 8, 9, 10], dtype=float)

    model = RegressionTree(max_depth=3)
    model.fit(X, y)

    r2 = model.score(X, y)
    assert r2 > 0.80


# ---------------------------------------------------------------------
# Hyperparameter Behavior
# ---------------------------------------------------------------------
def test_max_depth_effect():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([1, 1, 2, 8, 9, 10], dtype=float)

    shallow = RegressionTree(max_depth=1)
    deep = RegressionTree(max_depth=5)

    shallow.fit(X, y)
    deep.fit(X, y)

    assert deep.score(X, y) >= shallow.score(X, y)


def test_min_samples_leaf_constraint():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([1, 1, 2, 8, 9, 10], dtype=float)

    model = RegressionTree(min_samples_leaf=3)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert len(y_pred) == len(y)