import requests
from pathlib import Path

from source.data.api_fetcher.api_fetcher import ApiFetcher
from source.constants import DEFAULT_CHUNK_SIZE, DEFAULT_TIMEOUT_SECONDS


class HttpApiFetcher(ApiFetcher):
    """Streams a remote file to disk over HTTP(S) using requests."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}.")
        self._chunk_size: int = chunk_size
        self._timeout_seconds: int = timeout_seconds

    def fetch(self, url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True, timeout=self._timeout_seconds) as response:
            response.raise_for_status()
            with destination.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=self._chunk_size):
                    if chunk:
                        file_handle.write(chunk)

        return destination
