import numpy as np
from numpy.typing import ArrayLike, NDArray

from source.data.encoders.encoder import Encoder


class OneHotEncoder(Encoder):
    """One-hot encoder over a fixed, ordered set of classes declared at construction time."""

    def __init__(self, classes: ArrayLike) -> None:
        if classes is None or len(classes) < 2:
            raise ValueError("OneHotEncoder requires at least 2 classes.")
        self._classes = np.asarray(classes)

    @property
    def classes(self) -> NDArray:
        return self._classes

    @property
    def count_of_classes(self) -> int:
        return self._classes.shape[0]

    def encode(self, values: ArrayLike) -> NDArray:
        values_arr = np.asarray(values)
        unseen = np.setdiff1d(np.unique(values_arr), self._classes)
        if unseen.shape[0] > 0:
            raise ValueError(
                f"values contain labels not in declared classes: {unseen.tolist()}. "
                f"Declared classes: {self._classes.tolist()}"
            )

        count_of_samples = values_arr.shape[0]
        encoded = np.zeros((count_of_samples, self.count_of_classes), dtype=np.float64)
        # Set exactly one column per row to 1 — the column index matches the class's position in self._classes.
        for class_index, class_label in enumerate(self._classes):
            encoded[values_arr == class_label, class_index] = 1.0
        return encoded

    def decode(self, encoded: ArrayLike) -> NDArray:
        encoded_arr = np.asarray(encoded)
        if encoded_arr.ndim != 2 or encoded_arr.shape[1] != self.count_of_classes:
            raise ValueError(
                f"Expected encoded shape (n_samples, {self.count_of_classes}), "
                f"got {tuple(encoded_arr.shape)}."
            )
        # argmax also handles soft probability rows, not just hard one-hot vectors — useful for decoding predict_proba output.
        winning_indices = np.argmax(encoded_arr, axis=1)
        return self._classes[winning_indices]
