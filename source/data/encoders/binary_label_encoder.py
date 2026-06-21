import numpy as np
from numpy.typing import ArrayLike, NDArray

from source.data.encoders.encoder import Encoder
from source.constants import NEGATIVE_LABEL, POSITIVE_LABEL


class BinaryLabelEncoder(Encoder):
    """Maps a fixed pair of class labels to {-1, +1} for binary SVM training."""

    def __init__(self, classes: ArrayLike) -> None:
        classes_arr = np.asarray(classes)
        if classes_arr.shape[0] != 2:
            raise ValueError(
                f"BinaryLabelEncoder requires exactly 2 classes, got {classes_arr.shape[0]}."
            )
        if classes_arr[0] == classes_arr[1]:
            raise ValueError("BinaryLabelEncoder requires two distinct classes.")
        self._classes = classes_arr

    @property
    def classes(self) -> NDArray:
        return self._classes

    def encode(self, values: ArrayLike) -> NDArray:
        values_arr = np.asarray(values)
        unseen = np.setdiff1d(np.unique(values_arr), self._classes)
        if unseen.shape[0] > 0:
            raise ValueError(
                f"values contain labels not in declared classes: {unseen.tolist()}. "
                f"Declared classes: {self._classes.tolist()}"
            )

        encoded = np.empty(values_arr.shape[0], dtype=np.float64)
        # Class at index 0 is the negative side of the hyperplane, index 1 the positive side — order matters.
        encoded[values_arr == self._classes[0]] = NEGATIVE_LABEL
        encoded[values_arr == self._classes[1]] = POSITIVE_LABEL
        return encoded

    def decode(self, encoded: ArrayLike) -> NDArray:
        encoded_arr = np.asarray(encoded)
        if encoded_arr.ndim != 1:
            raise ValueError(f"Expected 1D encoded array, got shape {tuple(encoded_arr.shape)}.")

        decoded = np.empty(encoded_arr.shape[0], dtype=self._classes.dtype)
        # Threshold at zero — points on the boundary are deterministically assigned to the negative class.
        decoded[encoded_arr <= 0] = self._classes[0]
        decoded[encoded_arr > 0] = self._classes[1]
        return decoded
