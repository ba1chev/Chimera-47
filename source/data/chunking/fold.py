from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """Immutable container for one fold of a k-fold split: indices into the original arrays."""

    train_indices: NDArray
    validation_indices: NDArray

    @property
    def count_of_train_indices(self) -> int:
        return int(self.train_indices.shape[0])

    @property
    def count_of_validation_indices(self) -> int:
        return int(self.validation_indices.shape[0])
