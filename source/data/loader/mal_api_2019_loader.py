import numpy as np
from pathlib import Path
from numpy.typing import NDArray

from source.data.loader.dataset import Dataset
from source.data.loader.dataset_loader import DatasetLoader
from source.constants import SAMPLES_FILENAME, LABELS_FILENAME


class MalApi2019Loader(DatasetLoader):
    """Loads the Mal-API-2019 dataset (7107 Windows API call traces, 8 malware families)."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self._data_dir}")

    def load(self) -> Dataset:
        samples = self._load_samples()
        raw_labels = self._load_raw_labels()

        if samples.shape[0] != raw_labels.shape[0]:
            raise ValueError(
                f"Samples/labels length mismatch: {samples.shape[0]} samples vs {raw_labels.shape[0]} labels."
            )

        # np.unique with return_inverse gives integer-encoded labels and the canonical name table in one pass.
        label_names, labels = np.unique(raw_labels, return_inverse=True)
        return Dataset(samples=samples, labels=labels, label_names=label_names)

    def _load_samples(self) -> NDArray:
        path = self._data_dir / SAMPLES_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Samples file not found: {path}")

        with path.open("r", encoding="utf-8") as file_handle:
            lines = [line.strip() for line in file_handle if line.strip()]
        return np.asarray(lines, dtype=object)

    def _load_raw_labels(self) -> NDArray:
        path = self._data_dir / LABELS_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Labels file not found: {path}")

        with path.open("r", encoding="utf-8") as file_handle:
            lines = [line.strip() for line in file_handle if line.strip()]
        return np.asarray(lines, dtype=object)
