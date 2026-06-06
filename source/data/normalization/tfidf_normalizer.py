from typing import Iterable, Tuple
from numpy.typing import ArrayLike, NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from source.data.normalization.normalizer import Normalizer


class TfidfNormalizer(Normalizer):
    """TF-IDF + L2 normalization for API call traces."""

    def __init__(self, max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 1), sublinear_tf: bool = True) -> None:
        super().__init__()
        self._vectorizer: TfidfVectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",
            norm="l2",
            sublinear_tf=sublinear_tf,
            max_features=max_features,
            ngram_range=ngram_range
        )

    def fit(self, X: ArrayLike) -> "TfidfNormalizer":
        self._vectorizer.fit(self._as_iterable_of_strings(X))
        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform().")

        sparse_matrix = self._vectorizer.transform(self._as_iterable_of_strings(X))
        return sparse_matrix.toarray()

    @property
    def vocabulary(self) -> dict:
        if not self._is_fitted:
            raise RuntimeError("vocabulary is not available before the normalizer is fitted.")
        return self._vectorizer.vocabulary_

    @property
    def idf(self) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("idf is not available before the normalizer is fitted.")
        return self._vectorizer.idf_

    @staticmethod
    def _as_iterable_of_strings(X: ArrayLike) -> Iterable[str]:
        return [str(item) for item in X]
