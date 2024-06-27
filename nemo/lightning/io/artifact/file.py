import shutil
from pathlib import Path
from typing import Union

from nemo.lightning.io.artifact.base import Artifact


class PathArtifact(Artifact[Path]):
    def dump(self, value: Path, path: Path) -> Path:
        new_value = copy_file(value, path)
        return new_value

    def load(self, path: Path) -> Path:
        return path


class FileArtifact(Artifact[str]):
    def dump(self, value: str, path: Path) -> str:
        new_value = copy_file(value, path)
        return str(new_value)

    def load(self, path: str) -> str:
        return path


def copy_file(src: Union[Path, str], dst: Union[Path, str]):
    output = Path(dst) / Path(src).name
    shutil.copy2(src, output)
    return output
