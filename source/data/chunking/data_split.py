from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True)
class DataSplit:
    """Immutable container for a single train/test partition."""

    X_train: NDArray
    y_train: NDArray
    X_test: NDArray
    y_test: NDArray

    @property
    def count_of_train_samples(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def count_of_test_samples(self) -> int:
        return int(self.X_test.shape[0])
