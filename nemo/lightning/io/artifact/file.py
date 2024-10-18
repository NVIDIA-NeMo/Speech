import os
import shutil
from pathlib import Path
from typing import Union
import fiddle as fdl

from nemo.lightning.io.artifact.base import Artifact


class PathArtifact(Artifact[Path]):
    def dump(self, value: Path, absolute_dir: Path, relative_dir: Path) -> Path:
        new_value = copy_file(value, absolute_dir, relative_dir)
        return new_value

    def load(self, path: Path) -> Path:
        return path


class FileArtifact(Artifact[str]):
    def dump(self, value: str, absolute_dir: Path, relative_dir: Path) -> str:
        if not pathize(value).exists():
            # This is Artifact is just a string.
            return fdl.Config(FileArtifact, attr=value, skip=True)
        new_value = copy_file(value, absolute_dir, relative_dir)
        return str(new_value)

    def load(self, path: str) -> str:
        return path


def pathize(s):
    if not isinstance(s, Path):
        return Path(s)
    return s


def copy_file(src: Union[Path, str], path: Union[Path, str], relative_dst: Union[Path, str]):
    relative_path = pathize(relative_dst) / pathize(src).name
    output = pathize(path) / relative_path
    if output.exists():
        raise FileExistsError(f"Dst file already exists {str(output)}")
    shutil.copy2(src, output)
    return relative_path


class DirArtifact(Artifact[str]):
    def dump(self, value: str, absolute_dir: Path, relative_dir: Path) -> str:
        value = pathize(value)
        absolute_dir = pathize(absolute_dir)
        relative_dir = pathize(relative_dir)
        if not value.is_dir():
            return value

        relative_dir = relative_dir / value.name
        os.makedirs(str(absolute_dir / relative_dir), exist_ok=True)
        for file in value.iterdir():
            copy_file(file, absolute_dir, relative_dir)
        return str(relative_dir)

    def load(self, path: str) -> str:
        return path


class DirOrStringArtifact(DirArtifact):
    def dump(self, value: str, absolute_dir: Path, relative_dir: Path) -> str:
        if not pathize(value).exists():
            # This is Artifact is just a string.
            return fdl.Config(DirOrStringArtifact, attr=value, skip=True)
        return super().dump(value, absolute_dir, relative_dir)

    def load(self, path: str) -> str:
        return path
