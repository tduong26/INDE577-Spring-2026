import numpy as np

from rice_ml.supervised_ml.decision_tree import DecisionTreeClassifier


def test_simple_tree_fit_and_predict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X, y)

    y_pred = tree.predict(X)
    assert np.array_equal(y_pred, y)


def test_predict_proba_shape():
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 1, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    proba = tree.predict_proba(X)
    assert proba.shape == (3, 2)


def test_score_perfect_on_simple_data():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X, y)

    score = tree.score(X, y)
    assert np.isclose(score, 1.0)


def test_feature_importances_shape():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    tree.fit(X, y)

    assert tree.feature_importances_.shape == (2,)