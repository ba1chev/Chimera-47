import pytest
import numpy as np

from source.evaluations.confusion_matrix import ConfusionMatrix


class TestConfusionMatrix:
    def test_constructor_raises_on_single_class(self):
        with pytest.raises(ValueError, match="at least 2"):
            ConfusionMatrix(["only"])

    def test_constructor_raises_on_duplicate_classes(self):
        with pytest.raises(ValueError, match="distinct"):
            ConfusionMatrix(["a", "a"])

    def test_perfect_predictions_yield_diagonal_only(self):
        confusion_matrix = ConfusionMatrix(["a", "b", "c"])
        matrix = confusion_matrix.compute(
            np.array(["a", "b", "c", "a", "b"]), np.array(["a", "b", "c", "a", "b"])
        )
        np.testing.assert_array_equal(matrix, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    def test_off_diagonal_cells_count_misclassifications(self):
        confusion_matrix = ConfusionMatrix(["a", "b"])
        matrix = confusion_matrix.compute(
            np.array(["a", "a", "b", "b"]), np.array(["a", "b", "a", "b"])
        )
        np.testing.assert_array_equal(matrix, [[1, 1], [1, 1]])

    def test_row_sums_equal_true_class_counts(self):
        confusion_matrix = ConfusionMatrix(["a", "b", "c"])
        y_true = np.array(["a", "a", "a", "b", "b", "c"])
        y_predicted = np.array(["a", "b", "c", "b", "a", "c"])
        matrix = confusion_matrix.compute(y_true, y_predicted)
        np.testing.assert_array_equal(matrix.sum(axis=1), [3, 2, 1])

    def test_compute_raises_on_unknown_label_in_y_true(self):
        confusion_matrix = ConfusionMatrix(["a", "b"])
        with pytest.raises(ValueError, match="y_true"):
            confusion_matrix.compute(np.array(["a", "outsider"]), np.array(["a", "b"]))

    def test_compute_raises_on_unknown_label_in_y_predicted(self):
        confusion_matrix = ConfusionMatrix(["a", "b"])
        with pytest.raises(ValueError, match="y_predicted"):
            confusion_matrix.compute(np.array(["a", "b"]), np.array(["a", "outsider"]))

    def test_compute_raises_on_shape_mismatch(self):
        confusion_matrix = ConfusionMatrix(["a", "b"])
        with pytest.raises(ValueError, match="Shape mismatch"):
            confusion_matrix.compute(np.array(["a", "b", "a"]), np.array(["a", "b"]))

    def test_classes_property_preserves_declared_order(self):
        confusion_matrix = ConfusionMatrix(["c", "a", "b"])
        np.testing.assert_array_equal(confusion_matrix.classes, ["c", "a", "b"])
