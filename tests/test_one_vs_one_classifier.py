import pytest
import numpy as np

from source.data.encoders.binary_label_encoder import BinaryLabelEncoder
from source.models.support_vector_machine_model.soft_margin_svm import SoftMarginSVM
from source.models.one_vs_one_classifier.one_vs_one_classifier import OneVsOneClassifier


def make_four_class_dataset(samples_per_class: int = 30, seed: int = 47):
    rng = np.random.default_rng(seed)
    centers = [(-4, -4), (-4, 4), (4, -4), (4, 4)]
    labels = ["A", "B", "C", "D"]
    X = np.vstack([rng.normal(center, 0.4, (samples_per_class, 2)) for center in centers])
    y = np.array([label for label in labels for _ in range(samples_per_class)])
    return X, y, labels


def svm_factory(pair):
    encoder = BinaryLabelEncoder(list(pair))
    return SoftMarginSVM(
        classes=list(pair), binary_label_encoder=encoder,
        regularization_strength=1.0, learning_rate=0.05, max_iterations=1000
    )


class TestOneVsOneClassifier:
    def test_constructor_raises_on_none_factory(self):
        with pytest.raises(ValueError, match="binary_model_factory"):
            OneVsOneClassifier(classes=["a", "b", "c"], binary_model_factory=None)

    def test_models_count_is_combinations_of_two(self):
        ovo = OneVsOneClassifier(classes=["a", "b", "c", "d"], binary_model_factory=svm_factory)
        assert ovo.models_count == 6

    def test_models_count_for_eight_classes_is_twenty_eight(self):
        ovo = OneVsOneClassifier(
            classes=[f"c{i}" for i in range(8)], binary_model_factory=svm_factory
        )
        assert ovo.models_count == 28

    def test_predict_raises_before_fit(self):
        ovo = OneVsOneClassifier(classes=["a", "b"], binary_model_factory=svm_factory)
        with pytest.raises(RuntimeError, match="fitted"):
            ovo.predict(np.array([[1.0, 2.0]]))

    def test_fit_raises_when_a_declared_class_is_missing_from_y(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array(["A", "B"])
        ovo = OneVsOneClassifier(classes=["A", "B", "C"], binary_model_factory=svm_factory)
        with pytest.raises(ValueError, match="missing"):
            ovo.fit(X, y)

    def test_fit_classifies_four_separable_clusters_perfectly(self):
        X, y, labels = make_four_class_dataset()
        ovo = OneVsOneClassifier(classes=labels, binary_model_factory=svm_factory).fit(X, y)
        accuracy = float(np.mean(ovo.predict(X) == y))
        assert accuracy == 1.0

    def test_predictions_only_contain_declared_classes(self):
        X, y, labels = make_four_class_dataset()
        ovo = OneVsOneClassifier(classes=labels, binary_model_factory=svm_factory).fit(X, y)
        assert set(ovo.predict(X)).issubset(set(labels))

    def test_trains_one_model_per_class_pair(self):
        X, y, labels = make_four_class_dataset()
        ovo = OneVsOneClassifier(classes=labels, binary_model_factory=svm_factory).fit(X, y)
        assert len(ovo._models) == 6
