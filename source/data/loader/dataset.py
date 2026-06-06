from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """Immutable container for a loaded dataset."""

    samples: NDArray
    labels: NDArray
    label_names: NDArray

    @property
    def count_of_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def count_of_classes(self) -> int:
        return int(self.label_names.shape[0])
