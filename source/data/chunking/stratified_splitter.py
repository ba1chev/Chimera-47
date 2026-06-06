import numpy as np
from numpy.typing import ArrayLike, NDArray

from source.data.chunking.data_split import DataSplit
from source.data.chunking.data_splitter import DataSplitter


class StratifiedSplitter(DataSplitter):
    """Stratified train/test split: preserves the per-class proportion in both partitions."""

    def __init__(self, test_size: float = 0.2, random_state: int = 47) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
        self._test_size: float = test_size
        self._random_state: int = random_state

    def split(self, X: ArrayLike, y: ArrayLike) -> DataSplit:
        X_arr: NDArray = np.asarray(X)
        y_arr: NDArray = np.asarray(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}."
            )

        rng: np.random.Generator = np.random.default_rng(self._random_state)
        train_indices, test_indices = self._stratified_indices(y_arr, rng)

        return DataSplit(
            X_train=X_arr[train_indices],
            y_train=y_arr[train_indices],
            X_test=X_arr[test_indices],
            y_test=y_arr[test_indices]
        )

    def _stratified_indices(self, y: NDArray, rng: np.random.Generator) -> tuple[NDArray, NDArray]:
        train_chunks: list[NDArray] = []
        test_chunks: list[NDArray] = []

        for class_label in np.unique(y):
            class_indices: NDArray = np.where(y == class_label)[0]
            shuffled: NDArray = rng.permutation(class_indices)
            count_in_test: int = int(round(shuffled.shape[0] * self._test_size))
            test_chunks.append(shuffled[:count_in_test])
            train_chunks.append(shuffled[count_in_test:])

        train_indices: NDArray = rng.permutation(np.concatenate(train_chunks))
        test_indices: NDArray = rng.permutation(np.concatenate(test_chunks))
        return train_indices, test_indices
