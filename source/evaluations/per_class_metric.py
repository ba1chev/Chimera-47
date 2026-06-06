import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike, NDArray

from source.evaluations.metric import Metric


class PerClassMetric(Metric):
    """Base class for metrics whose macro form averages a per-class score across all observed classes."""

    def evaluate(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        y_true_arr = np.asarray(y_true)
        y_predicted_arr = np.asarray(y_predicted)
        if y_true_arr.shape != y_predicted_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true_arr.shape}, "
                f"y_predicted has shape {y_predicted_arr.shape}."
            )
        if y_true_arr.shape[0] == 0:
            raise ValueError("Cannot compute metric on empty arrays.")

        observed_classes = np.union1d(np.unique(y_true_arr), np.unique(y_predicted_arr))
        per_class_scores = [
            self._score_for_class(y_true_arr, y_predicted_arr, class_label)
            for class_label in observed_classes.tolist()
        ]
        return float(np.mean(per_class_scores))

    def _score_for_class(self, y_true: NDArray, y_predicted: NDArray, class_label) -> float:
        true_positive, false_positive, false_negative = self._counts_for_class(y_true, y_predicted, class_label)
        return self._score_from_counts(true_positive, false_positive, false_negative)

    @staticmethod
    def _counts_for_class(y_true: NDArray, y_predicted: NDArray, class_label) -> Tuple[int, int, int]:
        predicted_positive = y_predicted == class_label
        actually_positive = y_true == class_label
        true_positive = int(np.sum(predicted_positive & actually_positive))
        false_positive = int(np.sum(predicted_positive & ~actually_positive))
        false_negative = int(np.sum(~predicted_positive & actually_positive))
        return true_positive, false_positive, false_negative

    def _score_from_counts(self, true_positive: int, false_positive: int, false_negative: int) -> float:
        raise NotImplementedError("Subclasses must implement _score_from_counts")
