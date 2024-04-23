# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os
import tarfile
from typing import Union

import zarr.storage


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
        self._relpath = ''
        if isinstance(tar, TarPath):
            self._tar = tar._tar
            self._relpath = os.path.join(tar._relpath, *parts)
        elif isinstance(tar, tarfile.TarFile):
            self._tar = tar
            if parts:
                self._relpath = os.path.join(*parts)
        elif isinstance(tar, str):
            self._tar = tarfile.open(tar, 'r')
            if parts:
                self._relpath = os.path.join(*parts)
        else:
            raise ValueError(f"Unexpected argument type for TarPath: {type(tar).__name__}")

    def __truediv__(self, key) -> 'TarPath':
        return TarPath(self._tar, os.path.join(self._relpath, key))

    def __str__(self) -> str:
        return os.path.join(self._tar.name, self._relpath)

    @property
    def tarobject(self):
        return self._tar

    @property
    def relpath(self):
        return self._relpath

    @property
    def name(self):
        return os.path.split(self._relpath)[1]

    @property
    def suffix(self):
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
        try:
            self._tar.getmember(self._relpath).isdir()
            return True
        except KeyError:
            try:
                self._tar.getmember(os.path.join('.', self._relpath)).isdir()
                return True
            except KeyError:
                return False

    def open(self, mode: str):
        if mode != 'r' and mode != 'rb':
            raise NotImplementedError()
        try:
            # Try the relative path as-is first
            return self._tar.extractfile(self._relpath)
        except KeyError:
            try:
                # Try the relative path with "./" prefix
                return self._tar.extractfile(os.path.join('.', self._relpath))
            except KeyError:
                raise FileNotFoundError()

    def glob(self, pattern):
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
        return self.glob('*')


class ZarrPathStore(zarr.storage.BaseStore):
    """
    An implementation of read-only Store for zarr library
    that works with pathlib.Path or TarPath objects.
    """

    def __init__(self, tarpath: TarPath):
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
        return self._path.iterdir()


def unpack_tarball(archive: str, dest_dir: str):
    with tarfile.open(archive, mode="r") as tar:
        tar.extractall(path=dest_dir)
