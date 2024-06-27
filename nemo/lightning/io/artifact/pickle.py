from pathlib import Path
from typing import Any

from cloudpickle import dump, load

from nemo.lightning.io.artifact.base import Artifact


class PickleArtifact(Artifact[Any]):
    def dump(self, value: Any, path: Path) -> Path:
        file = self.file_path(path)
        with open(file, "wb") as f:
            dump(value, f)

        return file

    def load(self, path: Path) -> Any:
        with open(self.file_path(path), "rb") as f:
            return load(f)

    def file_path(self, path: Path) -> Path:
        return path / self.attr + ".pkl"
