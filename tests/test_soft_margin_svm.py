import pytest
import numpy as np

from source.data.encoders.binary_label_encoder import BinaryLabelEncoder
from source.models.support_vector_machine_model.soft_margin_svm import SoftMarginSVM


def make_separable_dataset(samples_per_class: int = 50, seed: int = 47):
    rng = np.random.default_rng(seed)
    class_a = rng.normal([-3.0, -3.0], 0.4, (samples_per_class, 2))
    class_b = rng.normal([3.0, 3.0], 0.4, (samples_per_class, 2))
    X = np.vstack([class_a, class_b])
    y = np.array(["neg"] * samples_per_class + ["pos"] * samples_per_class)
    return X, y


class TestSoftMarginSVM:
    def test_constructor_raises_on_three_classes(self):
        encoder = BinaryLabelEncoder(["a", "b"])
        with pytest.raises(ValueError, match="exactly 2"):
            SoftMarginSVM(["a", "b", "c"], encoder)

    def test_constructor_raises_on_none_encoder(self):
        with pytest.raises(ValueError, match="binary_label_encoder"):
            SoftMarginSVM(["a", "b"], None)

    def test_constructor_raises_on_non_positive_regularization_strength(self):
        encoder = BinaryLabelEncoder(["a", "b"])
        with pytest.raises(ValueError, match="regularization_strength"):
            SoftMarginSVM(["a", "b"], encoder, regularization_strength=0.0)

    def test_constructor_raises_on_non_positive_learning_rate(self):
        encoder = BinaryLabelEncoder(["a", "b"])
        with pytest.raises(ValueError, match="learning_rate"):
            SoftMarginSVM(["a", "b"], encoder, learning_rate=0.0)

    def test_decision_function_raises_before_fit(self):
        encoder = BinaryLabelEncoder(["a", "b"])
        model = SoftMarginSVM(["a", "b"], encoder)
        with pytest.raises(RuntimeError, match="fitted"):
            model.decision_function(np.array([[1.0, 2.0]]))

    def test_weights_property_raises_before_fit(self):
        encoder = BinaryLabelEncoder(["a", "b"])
        model = SoftMarginSVM(["a", "b"], encoder)
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.weights

    def test_fit_classifies_separable_data_perfectly(self):
        X, y = make_separable_dataset()
        encoder = BinaryLabelEncoder(["neg", "pos"])
        model = SoftMarginSVM(
            ["neg", "pos"], encoder, regularization_strength=1.0,
            learning_rate=0.05, max_iterations=2000
        ).fit(X, y)

        accuracy = float(np.mean(model.predict(X) == y))
        assert accuracy == 1.0

    def test_decision_function_sign_matches_predicted_class(self):
        X, y = make_separable_dataset()
        encoder = BinaryLabelEncoder(["neg", "pos"])
        model = SoftMarginSVM(
            ["neg", "pos"], encoder, regularization_strength=1.0,
            learning_rate=0.05, max_iterations=2000
        ).fit(X, y)

        scores = model.decision_function(X)
        predictions = model.predict(X)
        assert np.all((scores > 0) == (predictions == "pos"))

    def test_weights_have_one_value_per_feature(self):
        X, y = make_separable_dataset()
        encoder = BinaryLabelEncoder(["neg", "pos"])
        model = SoftMarginSVM(["neg", "pos"], encoder, max_iterations=200).fit(X, y)
        assert model.weights.shape == (X.shape[1],)

    def test_higher_regularization_strength_yields_larger_weights(self):
        X, y = make_separable_dataset()
        encoder_low = BinaryLabelEncoder(["neg", "pos"])
        encoder_high = BinaryLabelEncoder(["neg", "pos"])
        low_C = SoftMarginSVM(
            ["neg", "pos"], encoder_low, regularization_strength=0.01,
            learning_rate=0.01, max_iterations=2000
        ).fit(X, y)
        high_C = SoftMarginSVM(
            ["neg", "pos"], encoder_high, regularization_strength=10.0,
            learning_rate=0.01, max_iterations=2000
        ).fit(X, y)

        assert np.linalg.norm(high_C.weights) > np.linalg.norm(low_C.weights)
