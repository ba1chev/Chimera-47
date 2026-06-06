import numpy as np
from numpy.typing import ArrayLike, NDArray

from source.data.normalization.normalizer import Normalizer


class StandardNormalizer(Normalizer):
    """Z-score normalization: x' = (x - mu) / sigma. Learns mu, sigma per feature at fit time."""

    def __init__(self) -> None:
        super().__init__()
        self._mean: NDArray | None = None
        self._std: NDArray | None = None

    @staticmethod
    def _square_feature(feature: NDArray) -> NDArray:
        return feature ** 2

    @staticmethod
    def _get_mean_value(feature: NDArray) -> NDArray:
        return feature.mean(axis=0)

    def _get_std_value(self, feature: NDArray) -> NDArray:
        mean_of_squares: NDArray = self._get_mean_value(self._square_feature(feature))
        square_of_mean: NDArray = self._get_mean_value(feature) ** 2
        variance: NDArray = mean_of_squares - square_of_mean
        variance = np.maximum(variance, 0.0)
        return np.sqrt(variance)

    def fit(self, X: ArrayLike) -> "StandardNormalizer":
        X_arr: NDArray = np.asarray(X, dtype=np.float64)
        self._mean = self._get_mean_value(X_arr)
        self._std = self._get_std_value(X_arr)
        self._std = np.where(self._std == 0, 1.0, self._std)
        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform().")

        X_arr: NDArray = np.asarray(X, dtype=np.float64)
        return (X_arr - self._mean) / self._std
