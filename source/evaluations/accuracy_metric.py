import numpy as np
from numpy.typing import ArrayLike

from source.evaluations.metric import Metric


class AccuracyMetric(Metric):
    """Fraction of samples whose predicted label matches the true label."""

    def evaluate(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        y_true_arr = np.asarray(y_true)
        y_predicted_arr = np.asarray(y_predicted)
        if y_true_arr.shape != y_predicted_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true_arr.shape}, "
                f"y_predicted has shape {y_predicted_arr.shape}."
            )
        if y_true_arr.shape[0] == 0:
            raise ValueError("Cannot compute accuracy on empty arrays.")
        return float(np.mean(y_true_arr == y_predicted_arr))
