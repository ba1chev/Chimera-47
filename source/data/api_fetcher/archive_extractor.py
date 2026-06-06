from pathlib import Path
from abc import ABC, abstractmethod


class ArchiveExtractor(ABC):
    """Abstract base for extractors that unpack an archive into a destination directory."""

    @abstractmethod
    def extract(self, archive_path: Path, destination_dir: Path) -> Path:
        raise NotImplementedError("Must be implemented")
