from abc import ABC, abstractmethod

from typing import Any, Dict, Sized, Iterable, NamedTuple


DataEntry = Dict[str, Any]


class SourceContext(NamedTuple):
    source: str
    row: int


class Dataset(Sized, Iterable[DataEntry], ABC):
    @abstractmethod
    def __iter__(self) -> Iterable[DataEntry]:
        pass

    @abstractmethod
    def __len__(self):
        pass
