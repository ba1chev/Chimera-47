from typing import List
from numpy.typing import NDArray

from source.data.chunking.fold import Fold
from source.data.loader.dataset import Dataset
from source.data.chunking.data_split import DataSplit
from source.data.loader.dataset_loader import DatasetLoader
from source.data.normalization.normalizer import Normalizer
from source.data.chunking.data_splitter import DataSplitter
from source.data.chunking.k_fold_splitter import KFoldSplitter
from source.data.data_pipeline_result import DataPipelineResult
from source.data.api_fetcher.mal_api_2019_provisioner import MalApi2019Provisioner


class DataPipeline:
    """Orchestrates the full data path: provision -> load -> split -> normalize -> k-fold."""

    def __init__(self, provisioner: MalApi2019Provisioner, loader: DatasetLoader,
        normalizer: Normalizer, splitter: DataSplitter, k_fold_splitter: KFoldSplitter) -> None:
        self._provisioner: MalApi2019Provisioner = provisioner
        self._loader: DatasetLoader = loader
        self._normalizer: Normalizer = normalizer
        self._splitter: DataSplitter = splitter
        self._k_fold_splitter: KFoldSplitter = k_fold_splitter

    def run(self) -> DataPipelineResult:
        self._provisioner.provision()
        dataset: Dataset = self._loader.load()
        raw_split: DataSplit = self._splitter.split(dataset.samples, dataset.labels)
        X_train: NDArray = self._normalizer.fit_transform(raw_split.X_train)
        X_test: NDArray = self._normalizer.transform(raw_split.X_test)
        folds: List[Fold] = self._k_fold_splitter.split(X_train, raw_split.y_train)

        return DataPipelineResult(
            X_train=X_train,
            y_train=raw_split.y_train,
            X_test=X_test,
            y_test=raw_split.y_test,
            folds=folds,
            label_names=dataset.label_names
        )
