import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import StandardScaler

from source.data.normalization.normalizer import Normalizer


class StandardNormalizer(Normalizer):
    """Z-score normalization: x' = (x - mu) / sigma. Wraps sklearn StandardScaler."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler(with_mean=True, with_std=True)

    def fit(self, X: ArrayLike) -> "StandardNormalizer":
        X_arr = np.asarray(X, dtype=np.float64)
        self._scaler.fit(X_arr)
        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform().")

        X_arr = np.asarray(X, dtype=np.float64)
        return self._scaler.transform(X_arr)
