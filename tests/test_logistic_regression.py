import pytest
import numpy as np

from source.data.encoders.one_hot_encoder import OneHotEncoder
from source.models.logistic_regression_model.logistic_regression import LogisticRegression


def make_separable_two_class_dataset(samples_per_class: int = 50, seed: int = 47):
    rng = np.random.default_rng(seed)
    class_a = rng.normal(loc=[-3.0, -3.0], scale=0.5, size=(samples_per_class, 2))
    class_b = rng.normal(loc=[3.0, 3.0], scale=0.5, size=(samples_per_class, 2))
    X = np.vstack([class_a, class_b])
    y = np.array([0] * samples_per_class + [1] * samples_per_class)
    return X, y


def make_separable_three_class_dataset(samples_per_class: int = 40, seed: int = 47):
    rng = np.random.default_rng(seed)
    class_a = rng.normal(loc=[-4.0, 0.0], scale=0.4, size=(samples_per_class, 2))
    class_b = rng.normal(loc=[0.0, 4.0], scale=0.4, size=(samples_per_class, 2))
    class_c = rng.normal(loc=[4.0, 0.0], scale=0.4, size=(samples_per_class, 2))
    X = np.vstack([class_a, class_b, class_c])
    y = np.array([0] * samples_per_class + [1] * samples_per_class + [2] * samples_per_class)
    return X, y


class TestLogisticRegression:
    def test_constructor_raises_on_none_encoder(self):
        with pytest.raises(ValueError, match="one_hot_encoder"):
            LogisticRegression(classes=[0, 1], one_hot_encoder=None)

    def test_constructor_raises_on_non_positive_learning_rate(self):
        encoder = OneHotEncoder([0, 1])
        with pytest.raises(ValueError, match="learning_rate"):
            LogisticRegression(classes=[0, 1], one_hot_encoder=encoder, learning_rate=0.0)

    def test_constructor_raises_on_invalid_max_iterations(self):
        encoder = OneHotEncoder([0, 1])
        with pytest.raises(ValueError, match="max_iterations"):
            LogisticRegression(classes=[0, 1], one_hot_encoder=encoder, max_iterations=0)

    def test_constructor_raises_on_non_positive_tolerance(self):
        encoder = OneHotEncoder([0, 1])
        with pytest.raises(ValueError, match="tolerance"):
            LogisticRegression(classes=[0, 1], one_hot_encoder=encoder, tolerance=0.0)

    def test_predict_proba_raises_before_fit(self):
        encoder = OneHotEncoder([0, 1])
        model = LogisticRegression(classes=[0, 1], one_hot_encoder=encoder)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(np.array([[1.0, 2.0]]))

    def test_fit_classifies_separable_two_class_data(self):
        X, y = make_separable_two_class_dataset()
        encoder = OneHotEncoder([0, 1])
        model = LogisticRegression(
            classes=[0, 1], one_hot_encoder=encoder,
            learning_rate=0.1, max_iterations=2000, tolerance=1e-6
        )
        model.fit(X, y)

        predictions = model.predict(X)
        accuracy = float(np.mean(predictions == y))
        assert accuracy > 0.95

    def test_fit_classifies_separable_three_class_data(self):
        X, y = make_separable_three_class_dataset()
        encoder = OneHotEncoder([0, 1, 2])
        model = LogisticRegression(
            classes=[0, 1, 2], one_hot_encoder=encoder,
            learning_rate=0.1, max_iterations=2000, tolerance=1e-6
        )
        model.fit(X, y)

        predictions = model.predict(X)
        accuracy = float(np.mean(predictions == y))
        assert accuracy > 0.95

    def test_predict_proba_shape_and_sums_to_one(self):
        X, y = make_separable_three_class_dataset()
        encoder = OneHotEncoder([0, 1, 2])
        model = LogisticRegression(
            classes=[0, 1, 2], one_hot_encoder=encoder, max_iterations=200
        )
        model.fit(X, y)

        probabilities = model.predict_proba(X)
        assert probabilities.shape == (X.shape[0], 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)

    def test_predict_returns_declared_class_labels(self):
        X, y_indices = make_separable_two_class_dataset()
        labels = np.array(["malware", "benign"])
        y = labels[y_indices]
        encoder = OneHotEncoder(labels)
        model = LogisticRegression(
            classes=labels, one_hot_encoder=encoder,
            learning_rate=0.1, max_iterations=1000
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert set(np.unique(predictions)).issubset(set(labels))

    def test_fit_is_deterministic_for_same_data(self):
        X, y = make_separable_two_class_dataset()
        encoder_a = OneHotEncoder([0, 1])
        encoder_b = OneHotEncoder([0, 1])
        model_a = LogisticRegression(
            classes=[0, 1], one_hot_encoder=encoder_a,
            learning_rate=0.1, max_iterations=500, tolerance=1e-8
        ).fit(X, y)
        model_b = LogisticRegression(
            classes=[0, 1], one_hot_encoder=encoder_b,
            learning_rate=0.1, max_iterations=500, tolerance=1e-8
        ).fit(X, y)

        np.testing.assert_allclose(model_a.predict_proba(X), model_b.predict_proba(X))

    def test_fit_raises_on_undeclared_class_in_y(self):
        X, _ = make_separable_two_class_dataset()
        y_bad = np.array([0, 1, 99] * (X.shape[0] // 3) + [0] * (X.shape[0] - 3 * (X.shape[0] // 3)))
        encoder = OneHotEncoder([0, 1])
        model = LogisticRegression(classes=[0, 1], one_hot_encoder=encoder)

        with pytest.raises(ValueError, match="classes"):
            model.fit(X, y_bad)
