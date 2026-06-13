import pytest
import numpy as np

from source.models.logistic_regression.multinomial_logistic_regression import MultinomialLogisticRegression


def make_three_class_dataset(samples_per_class: int = 50, seed: int = 47):
    rng = np.random.default_rng(seed)
    centers = [(-4.0, -4.0), (4.0, -4.0), (0.0, 4.0)]
    labels = ["A", "B", "C"]
    X = np.vstack([rng.normal(center, 0.4, (samples_per_class, 2)) for center in centers])
    y = np.array([label for label in labels for _ in range(samples_per_class)])
    return X, y, labels


class TestMultinomialLogisticRegression:
    def test_constructor_raises_on_single_class(self):
        with pytest.raises(ValueError, match="at least 2 classes"):
            MultinomialLogisticRegression(["only"])

    def test_constructor_raises_on_non_positive_regularization_strength(self):
        with pytest.raises(ValueError, match="regularization_strength"):
            MultinomialLogisticRegression(["a", "b"], regularization_strength=0.0)

    def test_constructor_raises_on_non_positive_tolerance(self):
        with pytest.raises(ValueError, match="tolerance"):
            MultinomialLogisticRegression(["a", "b"], tolerance=0.0)

    def test_constructor_raises_on_zero_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            MultinomialLogisticRegression(["a", "b"], max_iterations=0)

    def test_predict_raises_before_fit(self):
        model = MultinomialLogisticRegression(["a", "b"])
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_predict_proba_raises_before_fit(self):
        model = MultinomialLogisticRegression(["a", "b"])
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(np.array([[1.0, 2.0]]))

    def test_weights_property_raises_before_fit(self):
        model = MultinomialLogisticRegression(["a", "b"])
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.weights

    def test_fit_raises_when_y_contains_undeclared_class(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array(["A", "Z"])
        model = MultinomialLogisticRegression(["A", "B"])
        with pytest.raises(ValueError, match="not declared"):
            model.fit(X, y)

    def test_fit_classifies_three_separable_clusters_perfectly(self):
        X, y, labels = make_three_class_dataset()
        model = MultinomialLogisticRegression(labels).fit(X, y)
        accuracy = float(np.mean(model.predict(X) == y))
        assert accuracy == 1.0

    def test_predictions_only_contain_declared_classes(self):
        X, y, labels = make_three_class_dataset()
        model = MultinomialLogisticRegression(labels).fit(X, y)
        assert set(model.predict(X)).issubset(set(labels))

    def test_predict_proba_returns_rows_summing_to_one(self):
        X, y, labels = make_three_class_dataset()
        model = MultinomialLogisticRegression(labels).fit(X, y)
        probabilities = model.predict_proba(X)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_shape_matches_classes_count(self):
        X, y, labels = make_three_class_dataset()
        model = MultinomialLogisticRegression(labels).fit(X, y)
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (X.shape[0], len(labels))

    def test_weights_have_one_row_per_class_one_column_per_feature(self):
        X, y, labels = make_three_class_dataset()
        model = MultinomialLogisticRegression(labels).fit(X, y)
        assert model.weights.shape == (len(labels), X.shape[1])
