import numpy as np
from typing import List
from numpy.typing import ArrayLike
from sklearn.model_selection import StratifiedKFold

from source.data.chunking.fold import Fold
from source.data.chunking.k_fold_splitter import KFoldSplitter
from source.constants import DEFAULT_COUNT_OF_FOLDS, DEFAULT_RANDOM_STATE


class StratifiedKFoldSplitter(KFoldSplitter):
    """Stratified k-fold splitter. Wraps sklearn StratifiedKFold."""

    def __init__(self, count_of_folds: int = DEFAULT_COUNT_OF_FOLDS, random_state: int = DEFAULT_RANDOM_STATE) -> None:
        if count_of_folds < 2:
            raise ValueError(f"count_of_folds must be >= 2, got {count_of_folds}.")
        self._count_of_folds = count_of_folds
        self._random_state = random_state

    def split(self, X: ArrayLike, y: ArrayLike) -> List[Fold]:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        splitter = StratifiedKFold(
            n_splits=self._count_of_folds,
            shuffle=True,
            random_state=self._random_state
        )

        folds = []
        for train_indices, validation_indices in splitter.split(X_arr, y_arr):
            folds.append(
                Fold(
                    train_indices=np.asarray(train_indices),
                    validation_indices=np.asarray(validation_indices)
                )
            )
        return folds
