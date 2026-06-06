import numpy as np
from typing import List
from numpy.typing import ArrayLike, NDArray

from source.data.chunking.fold import Fold
from source.data.chunking.k_fold_splitter import KFoldSplitter


class StratifiedKFoldSplitter(KFoldSplitter):
    """Stratified k-fold splitter: each fold's validation set preserves the per-class proportion."""

    def __init__(self, count_of_folds: int = 5, random_state: int = 47) -> None:
        if count_of_folds < 2:
            raise ValueError(f"count_of_folds must be >= 2, got {count_of_folds}.")
        self._count_of_folds: int = count_of_folds
        self._random_state: int = random_state

    def split(self, X: ArrayLike, y: ArrayLike) -> List[Fold]:
        y_arr: NDArray = np.asarray(y)
        count_of_samples: int = y_arr.shape[0]

        rng: np.random.Generator = np.random.default_rng(self._random_state)
        fold_assignments: NDArray = self._assign_folds(y_arr, rng)

        folds: List[Fold] = []
        for fold_id in range(self._count_of_folds):
            validation_mask: NDArray = (fold_assignments == fold_id)
            validation_indices: NDArray = np.where(validation_mask)[0]
            train_indices: NDArray = np.where(~validation_mask)[0]
            folds.append(
                Fold(train_indices=train_indices, validation_indices=validation_indices)
            )
        return folds

    def _assign_folds(self, y: NDArray, rng: np.random.Generator) -> NDArray:
        fold_assignments: NDArray = np.empty(y.shape[0], dtype=np.int64)

        for class_label in np.unique(y):
            class_indices: NDArray = np.where(y == class_label)[0]
            shuffled: NDArray = rng.permutation(class_indices)
            for position, sample_index in enumerate(shuffled):
                fold_assignments[sample_index] = position % self._count_of_folds

        return fold_assignments
