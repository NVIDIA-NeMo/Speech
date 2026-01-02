# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Union

import fiddle as fdl

from nemo.lightning.io.artifact.base import Artifact
from nemo.utils.file_utils import robust_copy


class PathArtifact(Artifact[Path]):
    """An artifact representing a `pathlib.Path` to a file."""

    def dump(self, instance, value: Path, absolute_dir: Path, relative_dir: Path) -> Path:
        new_value = copy_file(value, absolute_dir, relative_dir)
        return new_value

    def load(self, path: Path) -> Path:
        return path


class FileArtifact(Artifact[str]):
    """An artifact representing a file path as a string."""

    def dump(self, instance, value: str, absolute_dir: Path, relative_dir: Path) -> str:
        if not pathize(value).exists():
            # This is Artifact is just a string.
            return fdl.Config(FileArtifact, attr=value, skip=True)
        new_value = copy_file(value, absolute_dir, relative_dir)
        return str(new_value)

    def load(self, path: str) -> str:
        return path


def pathize(s):
    """Converts a string to a Path object if it is not already one."""
    if not isinstance(s, Path):
        return Path(s)
    return s


def copy_file(src: Union[Path, str], path: Union[Path, str], relative_dst: Union[Path, str]):
    """
    Copies a source file to a destination path within a relative directory structure.

    Args:
        src: The source file path.
        path: The absolute base path for the destination.
        relative_dst: The relative path within the base path.

    Returns:
        The relative path to the copied file.

    Raises:
        FileExistsError: If the destination file already exists.
    """
    relative_path = pathize(relative_dst) / pathize(src).name
    output = pathize(path) / relative_path
    if output.exists():
        raise FileExistsError(f"Dst file already exists {str(output)}")
    robust_copy(src, output)
    return relative_path


class DirArtifact(Artifact[str]):
    """An artifact representing a directory path as a string."""

    def dump(self, instance, value: str, absolute_dir: Path, relative_dir: Path) -> str:
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
    """An artifact that can be either a directory path or a plain string."""

    def dump(self, instance, value: str, absolute_dir: Path, relative_dir: Path) -> str:
        if not pathize(value).exists():
            # This is Artifact is just a string.
            return fdl.Config(DirOrStringArtifact, attr=value, skip=True)
        return super().dump(instance, value, absolute_dir, relative_dir)

    def load(self, path: str) -> str:
        return path
