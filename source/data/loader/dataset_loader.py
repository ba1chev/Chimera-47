from abc import ABC, abstractmethod

from source.data.loader.dataset import Dataset


class DatasetLoader(ABC):
    """Abstract base for dataset loaders. Subclasses define how raw files map to a Dataset."""

    @abstractmethod
    def load(self) -> Dataset:
        raise NotImplementedError("Must be implemented")
