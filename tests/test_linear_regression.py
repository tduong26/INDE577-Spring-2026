import numpy as np
import pytest

from rice_ml.supervised_ml.linear_regression import LinearRegression


def test_linear_regression_fit_and_predict_simple():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)  # y = 2x + 1

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == y.shape
    assert np.allclose(y_pred, y, atol=1e-8)


def test_linear_regression_intercept_and_coef():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)  # y = 2x + 1

    model = LinearRegression()
    model.fit(X, y)

    assert np.isclose(model.intercept_, 1.0, atol=1e-8)
    assert np.allclose(model.coef_, np.array([2.0]), atol=1e-8)


def test_linear_regression_score_perfect():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    assert np.isclose(r2, 1.0)


def test_linear_regression_residuals_zero_for_perfect_fit():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    residuals = model.residuals(X, y)
    assert np.allclose(residuals, np.zeros_like(y), atol=1e-8)


def test_linear_regression_error_metrics_zero_for_perfect_fit():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    assert np.isclose(model.mse(X, y), 0.0)
    assert np.isclose(model.rmse(X, y), 0.0)
    assert np.isclose(model.mae(X, y), 0.0)


def test_linear_regression_without_intercept():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 6, 8], dtype=float)  # y = 2x

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    assert np.isclose(model.intercept_, 0.0)
    assert np.allclose(model.coef_, np.array([2.0]), atol=1e-8)


def test_linear_regression_gradient_descent_fit():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)

    model = LinearRegression(
        use_gradient_descent=True,
        learning_rate=0.01,
        max_iter=10000,
        tol=1e-10,
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    assert np.allclose(y_pred, y, atol=1e-2)


def test_linear_regression_ridge_fit_runs():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)

    model = LinearRegression(regularization=1.0)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_linear_regression_length_mismatch_raises():
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([1, 2], dtype=float)

    model = LinearRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)