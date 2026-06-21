from pathlib import Path

from source.data.api_fetcher.api_fetcher import ApiFetcher
from source.data.api_fetcher.archive_extractor import ArchiveExtractor
from source.constants import SAMPLES_FILENAME, LABELS_FILENAME, ARCHIVE_FILENAME


class MalApi2019Provisioner:
    """Ensures the Mal-API-2019 dataset is present on disk: downloads, extracts, caches."""

    def __init__(self, data_dir: str, samples_archive_url: str, labels_url: str,
        fetcher: ApiFetcher, extractor: ArchiveExtractor) -> None:
        self._data_dir: Path = Path(data_dir)
        self._samples_archive_url: str = samples_archive_url
        self._labels_url: str = labels_url
        self._fetcher: ApiFetcher = fetcher
        self._extractor: ArchiveExtractor = extractor

    def provision(self) -> Path:
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Idempotent: each artefact is only downloaded if it isn't already cached on disk.
        labels_path: Path = self._data_dir / LABELS_FILENAME
        if not labels_path.is_file():
            self._fetcher.fetch(self._labels_url, labels_path)

        samples_path: Path = self._data_dir / SAMPLES_FILENAME
        if not samples_path.is_file():
            archive_path: Path = self._data_dir / ARCHIVE_FILENAME
            if not archive_path.is_file():
                self._fetcher.fetch(self._samples_archive_url, archive_path)
            self._extractor.extract(archive_path, self._data_dir)

        return self._data_dir
