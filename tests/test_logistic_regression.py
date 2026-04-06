import numpy as np
import pytest

from rice_ml.supervised_ml.logistic_regression import LogisticRegression


def test_sigmoid_outputs_between_zero_and_one():
    model = LogisticRegression()
    z = np.array([-1000, -1, 0, 1, 1000], dtype=float)

    probs = model.sigmoid(z)

    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_logistic_regression_fit_and_predict_shape():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=3000)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_logistic_regression_predict_proba_shape():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=3000)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (4,)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_logistic_regression_predict_binary_labels():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=3000)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_logistic_regression_score_high_on_simple_data():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=5000)
    model.fit(X, y)

    acc = model.score(X, y)
    assert acc >= 0.75


def test_logistic_regression_empty_input_raises():
    X = np.array([]).reshape(0, 1)
    y = np.array([])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_logistic_regression_length_mismatch_raises():
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 1], dtype=int)

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)