from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")


class Artifact(ABC, Generic[ValueT]):
    def __init__(self, attr: str):
        self.attr = attr

    @abstractmethod
    def dump(self, value: ValueT, path: Path) -> ValueT:
        pass

    @abstractmethod
    def load(self, path: Path) -> ValueT:
        pass
