import numpy as np
import pytest

from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


# ---------------------------------------------------------------------
# KNNClassifier Tests
# ---------------------------------------------------------------------
def test_knn_classifier_predict_basic():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=1, metric="euclidean", weights="uniform")
    clf.fit(X, y)

    y_pred = clf.predict([[0.1, 0.1]])
    assert y_pred[0] in [0, 1]

    probs = clf.predict_proba([[0.1, 0.1]])
    assert probs.shape == (1, 2)
    assert np.isclose(probs.sum(), 1.0)


def test_knn_classifier_predict_multiple():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=1)
    clf.fit(X, y)

    Xq = np.array([[0, 0], [1, 1]], dtype=float)
    y_pred = clf.predict(Xq)

    assert len(y_pred) == 2
    assert set(y_pred).issubset({0, 1})


def test_knn_classifier_score_perfect():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=1)
    clf.fit(X, y)

    score = clf.score(X, y)
    assert np.isclose(score, 1.0)


def test_knn_classifier_n_neighbors_exceeds_samples():
    X = np.array([[0], [1]], dtype=float)
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=3).fit(X, y)


def test_knn_classifier_manhattan_distance():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=1, metric="manhattan")
    clf.fit(X, y)

    probs = clf.predict_proba([[0.5, 0.5]])
    assert probs.shape == (1, 2)
    assert np.isclose(probs.sum(), 1.0)


# ---------------------------------------------------------------------
# KNNRegressor Tests
# ---------------------------------------------------------------------
def test_knn_regressor_predict_basic():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0.0, 1.0, 1.5, 3.0])

    reg_uniform = KNNRegressor(n_neighbors=2, weights="uniform")
    reg_uniform.fit(X, y)
    y_pred_uniform = reg_uniform.predict([[1.5]])
    assert 1.0 <= y_pred_uniform[0] <= 2.25

    reg_distance = KNNRegressor(n_neighbors=2, weights="distance")
    reg_distance.fit(X, y)
    y_pred_distance = reg_distance.predict([[1.5]])
    assert 1.0 <= y_pred_distance[0] <= 2.25


def test_knn_regressor_score_r2_perfect():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0.0, 1.0, 1.5, 3.0])

    reg = KNNRegressor(n_neighbors=1)
    reg.fit(X, y)

    score = reg.score(X, y)
    assert np.isclose(score, 1.0)


def test_knn_regressor_prediction_shape():
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0.0, 1.0, 2.0])

    reg = KNNRegressor(n_neighbors=2, weights="uniform")
    reg.fit(X, y)

    pred = reg.predict([[0.5], [1.5]])
    assert pred.shape == (2,)


def test_knn_regressor_invalid_targets():
    X = np.array([[0], [1]], dtype=float)
    y = np.array(["a", "b"])

    with pytest.raises(ValueError):
        KNNRegressor(n_neighbors=1).fit(X, y)