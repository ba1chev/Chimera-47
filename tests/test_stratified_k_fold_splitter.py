import pytest
import numpy as np

from source.data.chunking.stratified_k_fold_splitter import StratifiedKFoldSplitter


class TestStratifiedKFoldSplitter:
    def test_constructor_raises_on_one_fold(self):
        with pytest.raises(ValueError, match="count_of_folds"):
            StratifiedKFoldSplitter(count_of_folds=1)

    def test_returns_correct_number_of_folds(self):
        X = np.arange(50).reshape(-1, 1)
        y = np.repeat([0, 1, 2, 3, 4], 10)
        splitter = StratifiedKFoldSplitter(count_of_folds=5, random_state=47)
        folds = splitter.split(X, y)
        assert len(folds) == 5

    def test_validation_indices_partition_dataset(self):
        n_samples = 50
        X = np.arange(n_samples).reshape(-1, 1)
        y = np.repeat([0, 1, 2, 3, 4], 10)
        splitter = StratifiedKFoldSplitter(count_of_folds=5, random_state=47)
        folds = splitter.split(X, y)

        all_validation = np.concatenate([fold.validation_indices for fold in folds])
        np.testing.assert_array_equal(np.sort(all_validation), np.arange(n_samples))

    def test_train_and_validation_indices_are_disjoint(self):
        X = np.arange(50).reshape(-1, 1)
        y = np.repeat([0, 1, 2, 3, 4], 10)
        splitter = StratifiedKFoldSplitter(count_of_folds=5, random_state=47)
        folds = splitter.split(X, y)

        for fold in folds:
            train_set = set(fold.train_indices.tolist())
            validation_set = set(fold.validation_indices.tolist())
            assert train_set.isdisjoint(validation_set)

    def test_train_plus_validation_equals_dataset_size(self):
        X = np.arange(50).reshape(-1, 1)
        y = np.repeat([0, 1, 2, 3, 4], 10)
        splitter = StratifiedKFoldSplitter(count_of_folds=5, random_state=47)
        folds = splitter.split(X, y)

        for fold in folds:
            assert fold.train_indices.shape[0] + fold.validation_indices.shape[0] == 50

    def test_class_proportions_preserved_in_each_validation_fold(self):
        y = np.array([0] * 80 + [1] * 20)
        X = np.arange(100).reshape(-1, 1)
        splitter = StratifiedKFoldSplitter(count_of_folds=5, random_state=47)
        folds = splitter.split(X, y)

        for fold in folds:
            class_zero_count = (y[fold.validation_indices] == 0).sum()
            class_one_count = (y[fold.validation_indices] == 1).sum()
            total = class_zero_count + class_one_count
            assert class_zero_count / total == pytest.approx(0.8, abs=0.1)

    def test_same_random_state_produces_identical_folds(self):
        X = np.arange(50).reshape(-1, 1)
        y = np.repeat([0, 1, 2, 3, 4], 10)
        first = StratifiedKFoldSplitter(count_of_folds=5, random_state=47).split(X, y)
        second = StratifiedKFoldSplitter(count_of_folds=5, random_state=47).split(X, y)

        for fold_a, fold_b in zip(first, second):
            np.testing.assert_array_equal(fold_a.validation_indices, fold_b.validation_indices)
