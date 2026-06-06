import pytest
import zipfile
from pathlib import Path

from source.data.api_fetcher.zip_archive_extractor import ZipArchiveExtractor


class TestZipArchiveExtractor:
    def test_extract_raises_on_missing_archive(self, tmp_path):
        extractor = ZipArchiveExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract(tmp_path / "missing.zip", tmp_path)

    def test_extract_raises_on_invalid_zip(self, tmp_path):
        bad_archive: Path = tmp_path / "broken.zip"
        bad_archive.write_text("not a zip")
        extractor = ZipArchiveExtractor()
        with pytest.raises(ValueError, match="zip"):
            extractor.extract(bad_archive, tmp_path)

    def test_extract_creates_destination_directory(self, tmp_path):
        archive_path: Path = tmp_path / "data.zip"
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            zip_file.writestr("inside.txt", "hello")

        destination: Path = tmp_path / "out"
        extractor = ZipArchiveExtractor()
        extractor.extract(archive_path, destination)
        assert destination.is_dir()

    def test_extract_writes_archive_contents_to_destination(self, tmp_path):
        archive_path: Path = tmp_path / "data.zip"
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            zip_file.writestr("hello.txt", "world")

        destination: Path = tmp_path / "out"
        extractor = ZipArchiveExtractor()
        extractor.extract(archive_path, destination)
        assert (destination / "hello.txt").read_text() == "world"

    def test_extract_returns_destination_path(self, tmp_path):
        archive_path: Path = tmp_path / "data.zip"
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            zip_file.writestr("inside.txt", "x")

        destination: Path = tmp_path / "out"
        extractor = ZipArchiveExtractor()
        returned = extractor.extract(archive_path, destination)
        assert returned == destination
