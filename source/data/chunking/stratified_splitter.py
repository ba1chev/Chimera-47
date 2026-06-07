import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split

from source.data.chunking.data_split import DataSplit
from source.data.chunking.data_splitter import DataSplitter


class StratifiedSplitter(DataSplitter):
    """Stratified train/test split: preserves the per-class proportion. Wraps sklearn train_test_split."""

    def __init__(self, test_size: float = 0.2, random_state: int = 47) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
        self._test_size = test_size
        self._random_state = random_state

    def split(self, X: ArrayLike, y: ArrayLike) -> DataSplit:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y_arr,
            shuffle=True
        )

        return DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
