from pathlib import Path
from abc import ABC, abstractmethod


class ApiFetcher(ABC):
    """Abstract base for fetchers that download a single file from a remote URL to disk."""

    @abstractmethod
    def fetch(self, url: str, destination: Path) -> Path:
        raise NotImplementedError("Must be implemented")
