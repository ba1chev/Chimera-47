from typing import List
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

from source.data.chunking.fold import Fold


class KFoldSplitter(ABC):
    """Abstract base for k-fold splitters that emit a list of (train, validation) index folds."""

    @abstractmethod
    def split(self, X: ArrayLike, y: ArrayLike) -> List[Fold]:
        raise NotImplementedError("Must be implemented")
