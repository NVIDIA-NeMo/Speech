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

import fnmatch
import logging
import os
import tarfile

from typing import IO, Union

LOGGER = logging.getLogger("NeMo")

try:
    from zarr.storage import BaseStore

    HAVE_ZARR = True
except Exception as e:
    LOGGER.warning(f"Cannot import zarr, support for zarr-based checkpoints is not available. {type(e).__name__}: {e}")
    BaseStore = object
    HAVE_ZARR = False


class TarPath:
    """
    A class that represents a path inside a TAR archive and behaves like pathlib.Path.

    Expected use is to create a TarPath for the root of the archive first, and then derive
    paths to other files or directories inside the archive like so:

    with TarPath('/path/to/archive.tar') as archive:
        myfile = archive / 'filename.txt'
        if myfile.exists():
            data = myfile.read()
            ...

    Only read and enumeration operations are supported.
    """

    def __init__(self, tar: Union[str, tarfile.TarFile, 'TarPath'], *parts):
        self._needs_to_close = False
        self._relpath = ''
        if isinstance(tar, TarPath):
            self._tar = tar._tar
            self._relpath = os.path.join(tar._relpath, *parts)
        elif isinstance(tar, tarfile.TarFile):
            self._tar = tar
            if parts:
                self._relpath = os.path.join(*parts)
        elif isinstance(tar, str):
            self._needs_to_close = True
            self._tar = tarfile.open(tar, 'r')
            if parts:
                self._relpath = os.path.join(*parts)
        else:
            raise ValueError(f"Unexpected argument type for TarPath: {type(tar).__name__}")

    def __del__(self):
        if self._needs_to_close:
            self._tar.close()

    def __truediv__(self, key) -> 'TarPath':
        return TarPath(self._tar, os.path.join(self._relpath, key))

    def __str__(self) -> str:
        return os.path.join(self._tar.name, self._relpath)

    @property
    def tarobject(self):
        """
        Returns the wrapped tar object.
        """
        return self._tar

    @property
    def relpath(self):
        """
        Returns the relative path of the path.
        """
        return self._relpath

    @property
    def name(self):
        """
        Returns the name of the path.
        """
        return os.path.split(self._relpath)[1]

    @property
    def suffix(self):
        """
        Returns the suffix of the path.
        """
        name = self.name
        i = name.rfind('.')
        if 0 < i < len(name) - 1:
            return name[i:]
        else:
            return ''

    def __enter__(self):
        self._tar.__enter__()
        return self

    def __exit__(self, *args):
        return self._tar.__exit__(*args)

    def exists(self):
        """
        Checks if the path exists.
        """
        try:
            self._tar.getmember(self._relpath)
            return True
        except KeyError:
            try:
                self._tar.getmember(os.path.join('.', self._relpath))
                return True
            except KeyError:
                return False

    def is_file(self):
        """
        Checks if the path is a file.
        """
        try:
            self._tar.getmember(self._relpath).isreg()
            return True
        except KeyError:
            try:
                self._tar.getmember(os.path.join('.', self._relpath)).isreg()
                return True
            except KeyError:
                return False

    def is_dir(self):
        """
        Checks if the path is a directory.
        """
        try:
            self._tar.getmember(self._relpath).isdir()
            return True
        except KeyError:
            try:
                self._tar.getmember(os.path.join('.', self._relpath)).isdir()
                return True
            except KeyError:
                return False

    def open(self, mode: str) -> IO[bytes]:
        """
        Opens a file in the archive.
        """
        if mode != 'r' and mode != 'rb':
            raise NotImplementedError()

        file = None
        try:
            # Try the relative path as-is first
            file = self._tar.extractfile(self._relpath)
        except KeyError:
            try:
                # Try the relative path with "./" prefix
                file = self._tar.extractfile(os.path.join('.', self._relpath))
            except KeyError:
                raise FileNotFoundError()

        if file is None:
            raise FileNotFoundError()

        return file

    def glob(self, pattern):
        """
        Returns an iterator over the files in the directory, matching the pattern.
        """
        for member in self._tar.getmembers():
            # Remove the "./" prefix, if any
            name = member.name[2:] if member.name.startswith('./') else member.name

            # If we're in a subdirectory, make sure the file is too, and remove that subdir component
            if self._relpath:
                if not name.startswith(self._relpath + '/'):
                    continue
                name = name[len(self._relpath) + 1 :]

            # See if the name matches the pattern
            if fnmatch.fnmatch(name, pattern):
                yield TarPath(self._tar, os.path.join(self._relpath, name))

    def rglob(self, pattern):
        """
        Returns an iterator over the files in the directory, including subdirectories.
        """
        for member in self._tar.getmembers():
            # Remove the "./" prefix, if any
            name = member.name[2:] if member.name.startswith('./') else member.name

            # If we're in a subdirectory, make sure the file is too, and remove that subdir component
            if self._relpath:
                if not name.startswith(self._relpath + '/'):
                    continue
                name = name[len(self._relpath) + 1 :]

            # See if any tail of the path matches the pattern, return full path if that's true
            parts = name.split('/')
            for i in range(len(parts)):
                subname = '/'.join(parts[i:])
                if fnmatch.fnmatch(subname, pattern):
                    yield TarPath(self._tar, os.path.join(self._relpath, name))
                    break

    def iterdir(self):
        """
        Returns an iterator over the files in the directory.
        """
        return self.glob('*')


class ZarrPathStore(BaseStore):
    """
    An implementation of read-only Store for zarr library
    that works with pathlib.Path or TarPath objects.
    """

    def __init__(self, tarpath: TarPath):
        assert HAVE_ZARR, "Package zarr>=2.18.2,<3.0.0 is required to use ZarrPathStore"
        self._path = tarpath
        self._writable = False
        self._erasable = False

    def __getitem__(self, key):
        with (self._path / key).open('rb') as file:
            return file.read()

    def __contains__(self, key):
        return (self._path / key).is_file()

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def keys(self):
        """
        Returns an iterator over the keys in the store.
        """
        return self._path.iterdir()
