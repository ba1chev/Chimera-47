import pytest
import numpy as np

from source.data.chunking.stratified_splitter import StratifiedSplitter


class TestStratifiedSplitter:
    def test_constructor_raises_on_zero_test_size(self):
        with pytest.raises(ValueError, match="test_size"):
            StratifiedSplitter(test_size=0.0)

    def test_constructor_raises_on_one_test_size(self):
        with pytest.raises(ValueError, match="test_size"):
            StratifiedSplitter(test_size=1.0)

    def test_split_raises_on_length_mismatch(self):
        splitter = StratifiedSplitter()
        with pytest.raises(ValueError, match="length mismatch"):
            splitter.split(np.array([1, 2, 3]), np.array([0, 1]))

    def test_total_count_is_preserved(self):
        rng = np.random.RandomState(47)
        X = rng.randn(100, 5)
        y = rng.randint(0, 4, size=100)
        splitter = StratifiedSplitter(test_size=0.2, random_state=47)
        split = splitter.split(X, y)
        assert split.X_train.shape[0] + split.X_test.shape[0] == 100

    def test_test_size_is_respected_within_rounding(self):
        rng = np.random.RandomState(47)
        X = rng.randn(200, 3)
        y = rng.randint(0, 4, size=200)
        splitter = StratifiedSplitter(test_size=0.25, random_state=47)
        split = splitter.split(X, y)
        assert abs(split.X_test.shape[0] - 50) <= 4

    def test_class_proportions_are_preserved(self):
        y = np.array([0] * 80 + [1] * 20)
        X = np.arange(100).reshape(-1, 1)
        splitter = StratifiedSplitter(test_size=0.2, random_state=47)
        split = splitter.split(X, y)

        train_proportion_class_zero = (split.y_train == 0).sum() / split.y_train.shape[0]
        test_proportion_class_zero = (split.y_test == 0).sum() / split.y_test.shape[0]

        assert train_proportion_class_zero == pytest.approx(0.8, abs=0.05)
        assert test_proportion_class_zero == pytest.approx(0.8, abs=0.05)

    def test_train_and_test_indices_do_not_overlap(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.repeat([0, 1, 2, 3], 25)
        splitter = StratifiedSplitter(test_size=0.2, random_state=47)
        split = splitter.split(X, y)

        train_set = set(split.X_train.ravel().tolist())
        test_set = set(split.X_test.ravel().tolist())
        assert train_set.isdisjoint(test_set)

    def test_same_random_state_produces_identical_splits(self):
        rng = np.random.RandomState(47)
        X = rng.randn(50, 3)
        y = rng.randint(0, 3, size=50)

        first = StratifiedSplitter(test_size=0.2, random_state=47).split(X, y)
        second = StratifiedSplitter(test_size=0.2, random_state=47).split(X, y)
        np.testing.assert_array_equal(first.X_train, second.X_train)
        np.testing.assert_array_equal(first.y_test, second.y_test)

    def test_different_random_states_produce_different_splits(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        y = rng.randint(0, 3, size=50)

        first = StratifiedSplitter(test_size=0.2, random_state=1).split(X, y)
        second = StratifiedSplitter(test_size=0.2, random_state=2).split(X, y)
        assert not np.array_equal(first.X_train, second.X_train)
