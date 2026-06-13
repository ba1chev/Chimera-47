import pytest
import numpy as np

from source.models.markov_chain.markov_chain_classifier import MarkovChainClassifier


def generate_chain_sample(transition_probabilities, sequence_length, rng, vocabulary):
    vocabulary_size = len(vocabulary)
    initial_index = rng.integers(0, vocabulary_size)
    indices = [initial_index]
    for _ in range(sequence_length - 1):
        previous_index = indices[-1]
        next_index = rng.choice(vocabulary_size, p=transition_probabilities[previous_index])
        indices.append(next_index)
    return " ".join(vocabulary[index] for index in indices)


def make_two_chain_dataset(samples_per_class: int = 30, sequence_length: int = 60, seed: int = 47):
    rng = np.random.default_rng(seed)
    vocabulary = ["alpha", "beta", "gamma", "delta"]
    chain_a = np.array([
        [0.7, 0.2, 0.05, 0.05],
        [0.6, 0.3, 0.05, 0.05],
        [0.5, 0.4, 0.05, 0.05],
        [0.55, 0.35, 0.05, 0.05]
    ])
    chain_b = np.array([
        [0.05, 0.05, 0.4, 0.5],
        [0.05, 0.05, 0.3, 0.6],
        [0.05, 0.05, 0.7, 0.2],
        [0.05, 0.05, 0.55, 0.35]
    ])
    samples_a = [generate_chain_sample(chain_a, sequence_length, rng, vocabulary) for _ in range(samples_per_class)]
    samples_b = [generate_chain_sample(chain_b, sequence_length, rng, vocabulary) for _ in range(samples_per_class)]

    X = np.array(samples_a + samples_b, dtype=object)
    y = np.array(["A"] * samples_per_class + ["B"] * samples_per_class)
    return X, y


class TestMarkovChainClassifier:
    def test_constructor_raises_on_single_class(self):
        with pytest.raises(ValueError, match="at least 2 classes"):
            MarkovChainClassifier(["only"])

    def test_constructor_raises_on_non_positive_smoothing(self):
        with pytest.raises(ValueError, match="smoothing_alpha"):
            MarkovChainClassifier(["a", "b"], smoothing_alpha=0.0)

    def test_constructor_raises_on_negative_smoothing(self):
        with pytest.raises(ValueError, match="smoothing_alpha"):
            MarkovChainClassifier(["a", "b"], smoothing_alpha=-1.0)

    def test_predict_raises_before_fit(self):
        model = MarkovChainClassifier(["a", "b"])
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(np.array(["a b a"], dtype=object))

    def test_log_likelihoods_raises_before_fit(self):
        model = MarkovChainClassifier(["a", "b"])
        with pytest.raises(RuntimeError, match="fitted"):
            model.log_likelihoods(np.array(["a b a"], dtype=object))

    def test_vocabulary_size_raises_before_fit(self):
        model = MarkovChainClassifier(["a", "b"])
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.vocabulary_size

    def test_fit_raises_when_y_contains_undeclared_class(self):
        X = np.array(["a b a", "b a b"], dtype=object)
        y = np.array(["A", "Z"])
        model = MarkovChainClassifier(["A", "B"])
        with pytest.raises(ValueError, match="not declared"):
            model.fit(X, y)

    def test_fit_raises_on_length_mismatch(self):
        X = np.array(["a b a", "b a b"], dtype=object)
        y = np.array(["A"])
        model = MarkovChainClassifier(["A", "B"])
        with pytest.raises(ValueError, match="length mismatch"):
            model.fit(X, y)

    def test_classifies_two_distinguishable_chains_with_high_accuracy(self):
        X, y = make_two_chain_dataset()
        model = MarkovChainClassifier(["A", "B"]).fit(X, y)
        accuracy = float(np.mean(model.predict(X) == y))
        assert accuracy >= 0.95

    def test_predictions_only_contain_declared_classes(self):
        X, y = make_two_chain_dataset()
        model = MarkovChainClassifier(["A", "B"]).fit(X, y)
        assert set(model.predict(X)).issubset({"A", "B"})

    def test_log_likelihoods_have_expected_shape(self):
        X, y = make_two_chain_dataset(samples_per_class=10, sequence_length=20)
        model = MarkovChainClassifier(["A", "B"]).fit(X, y)
        likelihoods = model.log_likelihoods(X)
        assert likelihoods.shape == (X.shape[0], 2)

    def test_log_likelihood_is_higher_for_correct_class(self):
        X, y = make_two_chain_dataset(samples_per_class=20, sequence_length=80)
        model = MarkovChainClassifier(["A", "B"]).fit(X, y)
        likelihoods = model.log_likelihoods(X)
        for sample_index, class_label in enumerate(y):
            class_position = ["A", "B"].index(class_label)
            other_position = 1 - class_position
            assert likelihoods[sample_index, class_position] > likelihoods[sample_index, other_position]

    def test_unseen_token_in_test_does_not_crash(self):
        X_train = np.array(["a b a", "b a b", "a a b", "b b a"], dtype=object)
        y_train = np.array(["A", "A", "B", "B"])
        model = MarkovChainClassifier(["A", "B"]).fit(X_train, y_train)
        prediction = model.predict(np.array(["a c b a"], dtype=object))
        assert prediction[0] in {"A", "B"}

    def test_smoothing_prevents_negative_infinity(self):
        X_train = np.array(["a b a", "b a b"], dtype=object)
        y_train = np.array(["A", "B"])
        model = MarkovChainClassifier(["A", "B"], smoothing_alpha=1.0).fit(X_train, y_train)
        likelihoods = model.log_likelihoods(np.array(["a a a"], dtype=object))
        assert np.all(np.isfinite(likelihoods))

    def test_vocabulary_size_reflects_unique_training_tokens(self):
        X_train = np.array(["a b c", "a b", "c a"], dtype=object)
        y_train = np.array(["A", "B", "A"])
        model = MarkovChainClassifier(["A", "B"]).fit(X_train, y_train)
        assert model.vocabulary_size == 3

    def test_single_token_sequence_uses_initial_distribution_only(self):
        X_train = np.array(["a b a", "b a b", "a a b", "b b a"], dtype=object)
        y_train = np.array(["A", "A", "B", "B"])
        model = MarkovChainClassifier(["A", "B"]).fit(X_train, y_train)
        likelihoods = model.log_likelihoods(np.array(["a"], dtype=object))
        assert np.all(np.isfinite(likelihoods))
