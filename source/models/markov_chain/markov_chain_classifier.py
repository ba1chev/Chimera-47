import numpy as np
from typing import Dict, List
from numpy.typing import ArrayLike, NDArray

from source.models.supervised_learning_model import SupervisedLearningModel
from source.constants import UNKNOWN_TOKEN_INDEX, DEFAULT_SMOOTHING_ALPHA


class MarkovChainClassifier(SupervisedLearningModel):
    """Generative classifier — fits one first-order Markov chain per class over API token sequences."""

    def __init__(self, classes: ArrayLike, smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA) -> None:
        super().__init__(classes)
        if smoothing_alpha <= 0:
            raise ValueError(f"smoothing_alpha must be positive, got {smoothing_alpha}.")
        self._smoothing_alpha = smoothing_alpha
        self._token_to_index: Dict[str, int] = {}
        self._vocabulary_size: int = 0
        self._initial_log_probabilities: Dict[int, NDArray] = {}
        self._transition_log_probabilities: Dict[int, NDArray] = {}

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MarkovChainClassifier":
        X_arr = np.asarray(X, dtype=object)
        y_arr = self._validate_y(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}."
            )

        tokenized_sequences = [self._tokenize(sample) for sample in X_arr]
        self._build_vocabulary(tokenized_sequences)
        index_sequences = [self._encode_sequence(tokens) for tokens in tokenized_sequences]

        # One independent chain per class — the generative model is P(class) * P(seq | class).
        for class_label in self._classes.tolist():
            class_mask = y_arr == class_label
            class_sequences = [index_sequences[i] for i in np.where(class_mask)[0]]
            self._fit_single_chain(class_label, class_sequences)

        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict().")

        X_arr = np.asarray(X, dtype=object)
        scores = self._score(X_arr)
        # argmax over class log-likelihoods — equivalent to MAP with a uniform class prior.
        winning_indices = scores.argmax(axis=1)
        return self._classes[winning_indices]

    def log_likelihoods(self, X: ArrayLike) -> NDArray:
        """Per-class log-likelihoods for each sample. Shape: (count_of_samples, count_of_classes)."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before log_likelihoods().")
        X_arr = np.asarray(X, dtype=object)
        return self._score(X_arr)

    @property
    def vocabulary_size(self) -> int:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing vocabulary_size.")
        return self._vocabulary_size

    def _score(self, X_arr: NDArray) -> NDArray:
        count_of_samples = X_arr.shape[0]
        count_of_classes = self.count_of_classes()
        scores = np.empty((count_of_samples, count_of_classes), dtype=np.float64)

        for sample_index in range(count_of_samples):
            tokens = self._tokenize(X_arr[sample_index])
            indices = self._encode_sequence(tokens)
            for class_position, class_label in enumerate(self._classes.tolist()):
                scores[sample_index, class_position] = self._sequence_log_likelihood(class_label, indices)
        return scores

    def _sequence_log_likelihood(self, class_label, indices: NDArray) -> float:
        if indices.shape[0] == 0:
            return 0.0

        initial_log = self._initial_log_probabilities[class_label][indices[0]]
        if indices.shape[0] == 1:
            # Single-token sequence has no transitions — only the initial probability contributes.
            return float(initial_log)

        # log P(seq) = log P(x_0) + sum_t log P(x_t | x_{t-1}) — the chain factorisation in log space.
        transition_log = self._transition_log_probabilities[class_label]
        from_indices = indices[:-1]
        to_indices = indices[1:]
        # Vectorised lookup: gathers log P(x_t | x_{t-1}) for every consecutive (from, to) pair.
        transition_sum = float(transition_log[from_indices, to_indices].sum())
        return float(initial_log) + transition_sum

    def _fit_single_chain(self, class_label, sequences: List[NDArray]) -> None:
        # +1 for the reserved UNKNOWN_TOKEN_INDEX slot at index 0.
        vocabulary_with_unk = self._vocabulary_size + 1

        initial_counts = np.zeros(vocabulary_with_unk, dtype=np.float64)
        transition_counts = np.zeros((vocabulary_with_unk, vocabulary_with_unk), dtype=np.float64)

        for indices in sequences:
            if indices.shape[0] == 0:
                continue
            initial_counts[indices[0]] += 1.0
            if indices.shape[0] > 1:
                from_indices = indices[:-1]
                to_indices = indices[1:]
                # np.add.at supports unbuffered increments — needed when the same (from, to) pair appears twice.
                np.add.at(transition_counts, (from_indices, to_indices), 1.0)

        # Laplace smoothing: add alpha to every count so unseen events get non-zero probability.
        smoothed_initial = initial_counts + self._smoothing_alpha
        initial_probabilities = smoothed_initial / smoothed_initial.sum()
        self._initial_log_probabilities[class_label] = np.log(initial_probabilities)

        smoothed_transitions = transition_counts + self._smoothing_alpha
        # Row-wise normalisation: each row must sum to 1 since it is a conditional distribution P(x_t | x_{t-1}).
        row_sums = smoothed_transitions.sum(axis=1, keepdims=True)
        transition_probabilities = smoothed_transitions / row_sums
        self._transition_log_probabilities[class_label] = np.log(transition_probabilities)

    def _build_vocabulary(self, tokenized_sequences: List[List[str]]) -> None:
        unique_tokens = set()
        for tokens in tokenized_sequences:
            unique_tokens.update(tokens)
        sorted_tokens = sorted(unique_tokens)
        # Indices start at 1 because index 0 is reserved for UNKNOWN_TOKEN_INDEX.
        self._token_to_index = {token: index + 1 for index, token in enumerate(sorted_tokens)}
        self._vocabulary_size = len(sorted_tokens)

    def _encode_sequence(self, tokens: List[str]) -> NDArray:
        if not tokens:
            return np.empty(0, dtype=np.int64)
        indices = [self._token_to_index.get(token, UNKNOWN_TOKEN_INDEX) for token in tokens]
        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def _tokenize(sample) -> List[str]:
        if isinstance(sample, (list, tuple, np.ndarray)):
            return [str(token) for token in sample]
        return str(sample).split()
