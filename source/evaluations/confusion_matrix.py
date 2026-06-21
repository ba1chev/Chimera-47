import numpy as np
from numpy.typing import ArrayLike, NDArray


class ConfusionMatrix:
    """K x K confusion matrix where rows are true classes and columns are predicted classes."""

    def __init__(self, classes: ArrayLike) -> None:
        classes_arr = np.asarray(classes)
        if classes_arr.shape[0] < 2:
            raise ValueError(
                f"ConfusionMatrix requires at least 2 classes, got {classes_arr.shape[0]}."
            )
        if np.unique(classes_arr).shape[0] != classes_arr.shape[0]:
            raise ValueError("ConfusionMatrix requires distinct classes.")
        self._classes = classes_arr
        self._class_to_index = {label: index for index, label in enumerate(classes_arr.tolist())}

    @property
    def classes(self) -> NDArray:
        return self._classes

    def compute(self, y_true: ArrayLike, y_predicted: ArrayLike) -> NDArray:
        y_true_arr = np.asarray(y_true)
        y_predicted_arr = np.asarray(y_predicted)
        if y_true_arr.shape != y_predicted_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true_arr.shape}, "
                f"y_predicted has shape {y_predicted_arr.shape}."
            )

        count_of_classes = self._classes.shape[0]
        matrix = np.zeros((count_of_classes, count_of_classes), dtype=np.int64)
        # Diagonal entries are correct predictions; off-diagonal cells reveal which classes get confused.
        for true_label, predicted_label in zip(y_true_arr.tolist(), y_predicted_arr.tolist()):
            row = self._lookup(true_label, "y_true")
            column = self._lookup(predicted_label, "y_predicted")
            matrix[row, column] += 1
        return matrix

    def _lookup(self, label, source: str) -> int:
        if label not in self._class_to_index:
            raise ValueError(f"{source} contains label {label!r} not in declared classes.")
        return self._class_to_index[label]
