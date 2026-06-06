import zipfile
from pathlib import Path

from source.data.api_fetcher.archive_extractor import ArchiveExtractor


class ZipArchiveExtractor(ArchiveExtractor):
    """Unpacks .zip archives into a destination directory."""

    def extract(self, archive_path: Path, destination_dir: Path) -> Path:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive does not exist: {archive_path}")
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"Not a valid zip archive: {archive_path}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zip_handle:
            zip_handle.extractall(destination_dir)

        return destination_dir
