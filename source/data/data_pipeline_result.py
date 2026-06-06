from typing import List
from dataclasses import dataclass
from numpy.typing import NDArray

from source.data.chunking.fold import Fold


@dataclass(frozen=True)
class DataPipelineResult:
    """Immutable container holding everything the training stage needs."""

    X_train: NDArray
    y_train: NDArray
    X_test: NDArray
    y_test: NDArray
    folds: List[Fold]
    label_names: NDArray

    @property
    def count_of_train_samples(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def count_of_test_samples(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def count_of_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def count_of_classes(self) -> int:
        return int(self.label_names.shape[0])

    @property
    def count_of_folds(self) -> int:
        return len(self.folds)
