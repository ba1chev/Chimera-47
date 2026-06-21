import numpy as np
from typing import Callable, Tuple
from numpy.typing import ArrayLike, NDArray

from source.models.supervised_learning_model import SupervisedLearningModel


class OneVsOneClassifier(SupervisedLearningModel):
    """Multi-class classifier that delegates to one binary model per ordered class pair (i, j) with i < j."""

    def __init__(self, classes: ArrayLike,
        binary_model_factory: Callable[[Tuple], SupervisedLearningModel]) -> None:
        super().__init__(classes)
        if binary_model_factory is None:
            raise ValueError("binary_model_factory must not be None.")

        self._binary_model_factory = binary_model_factory
        # Ordered (i, j) with i < j — for K classes this is exactly C(K, 2) = K*(K-1)/2 binary problems.
        self._class_pairs = [
            (i, j)
            for i in range(self.count_of_classes())
            for j in range(i + 1, self.count_of_classes())
        ]
        self._models = {}

    @property
    def models_count(self) -> int:
        return len(self._class_pairs)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "OneVsOneClassifier":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = self._validate_y(y)
        self._require_every_class_present(y_arr)

        for class_index_a, class_index_b in self._class_pairs:
            class_a = self._classes[class_index_a]
            class_b = self._classes[class_index_b]
            # Train each binary model only on the rows belonging to its own class pair.
            pair_mask = (y_arr == class_a) | (y_arr == class_b)
            X_pair = X_arr[pair_mask]
            y_pair = y_arr[pair_mask]

            model = self._binary_model_factory((class_a, class_b))
            model.fit(X_pair, y_pair)
            self._models[(class_index_a, class_index_b)] = model

        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        X_arr = np.asarray(X, dtype=np.float64)
        count_of_samples = X_arr.shape[0]
        count_of_classes = self.count_of_classes()

        votes = np.zeros((count_of_samples, count_of_classes), dtype=np.int64)
        margin_sums = np.zeros((count_of_samples, count_of_classes), dtype=np.float64)

        for (class_index_a, class_index_b), model in self._models.items():
            pair_predictions = model.predict(X_arr)
            class_a = self._classes[class_index_a]
            votes[pair_predictions == class_a, class_index_a] += 1
            votes[pair_predictions == self._classes[class_index_b], class_index_b] += 1

            # Margin sum becomes the tiebreaker when two classes tie on vote count.
            decision_function = getattr(model, "decision_function", None)
            if decision_function is not None:
                scores = decision_function(X_arr)
                margin_sums[:, class_index_b] += scores
                margin_sums[:, class_index_a] -= scores

        winning_indices = self._resolve_winners(votes, margin_sums)
        return self._classes[winning_indices]

    def _resolve_winners(self, votes: NDArray, margin_sums: NDArray) -> NDArray:
        max_votes = votes.max(axis=1, keepdims=True)
        tie_mask = votes == max_votes
        # Non-tied classes get -inf so argmax can only pick among the tied ones.
        masked_margins = np.where(tie_mask, margin_sums, -np.inf)
        return masked_margins.argmax(axis=1)

    def _require_every_class_present(self, y_arr: NDArray) -> None:
        observed = np.unique(y_arr)
        missing = np.setdiff1d(self._classes, observed)
        if missing.shape[0] > 0:
            raise ValueError(
                f"y is missing samples for declared classes: {missing.tolist()}. "
                f"Every declared class must appear in y for one-vs-one training."
            )
