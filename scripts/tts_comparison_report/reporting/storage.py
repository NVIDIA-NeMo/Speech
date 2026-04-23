# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import errno
import json
from abc import ABC, abstractmethod
from pathlib import Path
from stat import S_ISDIR
from typing import Any, BinaryIO, Generator

from paramiko.sftp_client import SFTPClient


class BaseStorage(ABC):
    """Abstract storage interface for accessing report artifacts."""

    @abstractmethod
    def exists(self, path: Path) -> bool:
        """Check whether a path exists.

        Args:
            path: Path to check.

        Returns:
            True if the path exists, otherwise False.
        """
        ...

    @abstractmethod
    def iter_dir(
        self,
        path: Path,
        only_dirs: bool = False,
    ) -> Generator[Path, None, None]:
        """Iterate over items in a directory.

        Args:
            path: Directory path to iterate.
            only_dirs: Whether to yield only directory entries.

        Yields:
            Paths of directory entries.
        """
        ...

    @abstractmethod
    def open_file(self, path: Path) -> BinaryIO:
        """Open a file for binary reading.

        Args:
            path: Path to the file.

        Returns:
            Binary file-like object.
        """
        ...

    @abstractmethod
    def read_json(self, path: Path) -> Any:
        """Read and parse a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed JSON content.
        """
        ...

    @abstractmethod
    def read_bytes(self, path: Path) -> bytes:
        """Read raw bytes from a file in storage.

        Args:
            path: Path to the file.

        Returns:
            File content as bytes.
        """
        ...


class LocalStorage(BaseStorage):
    """Storage backend for accessing artifacts on the local filesystem."""

    def exists(self, path: Path) -> bool:
        """See the BaseStorage class docstring."""
        return path.exists()

    def iter_dir(
        self,
        path: Path,
        only_dirs: bool = False,
    ) -> Generator[Path, None, None]:
        """See the BaseStorage class docstring."""
        for p in path.iterdir():
            if only_dirs and not p.is_dir():
                continue
            yield p

    def open_file(self, path: Path) -> BinaryIO:
        """See the BaseStorage class docstring."""
        return open(path, "rb")

    def read_json(self, path: Path) -> Any:
        """See the BaseStorage class docstring."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def read_bytes(self, path: Path) -> bytes:
        """See the BaseStorage class docstring."""
        return path.read_bytes()


class SFTPStorage(BaseStorage):
    """Storage backend for accessing artifacts on a remote host over SFTP."""

    def __init__(self, sftp: SFTPClient) -> None:
        super().__init__()
        self.sftp = sftp

    def exists(self, path: Path) -> bool:
        """See the BaseStorage class docstring."""
        try:
            self.sftp.stat(path.as_posix())
            return True

        except FileNotFoundError:
            return False

        except OSError as e:
            if getattr(e, "errno", None) == errno.ENOENT:
                return False
            raise

    def iter_dir(
        self,
        path: Path,
        only_dirs: bool = False,
    ) -> Generator[Path, None, None]:
        """See the BaseStorage class docstring."""
        for item in self.sftp.listdir_attr(path.as_posix()):
            if only_dirs and not S_ISDIR(item.st_mode):
                continue
            yield path / item.filename

    def open_file(self, path: Path) -> BinaryIO:
        """See the BaseStorage class docstring."""
        return self.sftp.open(path.as_posix(), "rb")

    def read_json(self, path: Path) -> Any:
        """See the BaseStorage class docstring."""
        with self.sftp.open(path.as_posix(), "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        return data

    def read_bytes(self, path: Path) -> bytes:
        """See the BaseStorage class docstring."""
        with self.sftp.open(path.as_posix(), "rb") as f:
            data = f.read()
        return data
