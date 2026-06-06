import numpy as np

from source.data.encoders.binary_label_encoder import BinaryLabelEncoder
from source.models.support_vector_machine_model.hard_margin_svm import HardMarginSVM
from source.models.support_vector_machine_model.soft_margin_svm import SoftMarginSVM


def make_separable_dataset(samples_per_class: int = 50, seed: int = 47):
    rng = np.random.default_rng(seed)
    class_a = rng.normal([-3.0, -3.0], 0.4, (samples_per_class, 2))
    class_b = rng.normal([3.0, 3.0], 0.4, (samples_per_class, 2))
    X = np.vstack([class_a, class_b])
    y = np.array(["neg"] * samples_per_class + ["pos"] * samples_per_class)
    return X, y


class TestHardMarginSVM:
    def test_is_a_soft_margin_svm(self):
        encoder = BinaryLabelEncoder(["a", "b"])
        model = HardMarginSVM(["a", "b"], encoder)
        assert isinstance(model, SoftMarginSVM)

    def test_uses_very_large_regularization_strength(self):
        encoder = BinaryLabelEncoder(["a", "b"])
        model = HardMarginSVM(["a", "b"], encoder)
        assert model._regularization_strength >= 1e6

    def test_fit_classifies_separable_data_perfectly(self):
        X, y = make_separable_dataset()
        encoder = BinaryLabelEncoder(["neg", "pos"])
        model = HardMarginSVM(
            ["neg", "pos"], encoder,
            learning_rate=1e-5, max_iterations=5000
        ).fit(X, y)

        accuracy = float(np.mean(model.predict(X) == y))
        assert accuracy == 1.0

    def test_drives_all_separable_points_outside_unit_margin(self):
        X, y = make_separable_dataset()
        encoder = BinaryLabelEncoder(["neg", "pos"])
        model = HardMarginSVM(
            ["neg", "pos"], encoder,
            learning_rate=1e-5, max_iterations=5000
        ).fit(X, y)

        scores = model.decision_function(X)
        signed_y = encoder.encode(y)
        margins = signed_y * scores
        assert np.all(margins >= 0.99)
