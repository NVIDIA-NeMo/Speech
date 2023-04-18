# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import pathlib
import shutil
import subprocess
from typing import Tuple

from nemo import __version__ as NEMO_VERSION
from nemo import constants
from nemo.utils import logging


def resolve_cache_dir() -> pathlib.Path:
    """
    Utility method to resolve a cache directory for NeMo that can be overriden by an environment variable.

    Example:
        NEMO_CACHE_DIR="~/nemo_cache_dir/" python nemo_example_script.py

    Returns:
        A Path object, resolved to the absolute path of the cache directory. If no override is provided,
        uses an inbuilt default which adapts to nemo versions strings.
    """
    override_dir = os.environ.get(constants.NEMO_ENV_CACHE_DIR, "")
    if override_dir == "":
        path = pathlib.Path.joinpath(pathlib.Path.home(), f'.cache/torch/NeMo/NeMo_{NEMO_VERSION}')
    else:
        path = pathlib.Path(override_dir).resolve()
    return path


def is_datastore_path(path) -> bool:
    """Check if a path is from a data object store.
    Currently, only AIStore is supported.
    """
    return path.startswith('ais://')


def is_tarred_path(path) -> bool:
    """Check if a path is for a tarred file.
    """
    return path.endswith('.tar')


def is_datastore_cache_shared() -> bool:
    """Check if store cache is shared.
    """
    # Assume cache is shared by default, e.g., as in resolve_cache_dir (~/.cache)
    cache_shared = int(os.environ.get(constants.NEMO_ENV_DATA_STORE_CACHE_SHARED, 1))

    if cache_shared == 0:
        return False
    elif cache_shared == 1:
        return True
    else:
        raise ValueError(f'Unexpected value of env {constants.NEMO_ENV_DATA_STORE_CACHE_SHARED}')


def ais_cache_base() -> str:
    """Return path to local cache for AIS.
    """
    override_dir = os.environ.get(constants.NEMO_ENV_DATA_STORE_CACHE_DIR, "")
    if override_dir == "":
        cache_dir = resolve_cache_dir().as_posix()
    else:
        cache_dir = pathlib.Path(override_dir).resolve().as_posix()

    if cache_dir.endswith(NEMO_VERSION):
        # Prevent re-caching dataset after upgrading NeMo
        cache_dir = os.path.dirname(cache_dir)
    return os.path.join(cache_dir, 'ais')


def ais_endpoint() -> str:
    """Get configured AIS endpoint.
    """
    return os.getenv('AIS_ENDPOINT')


def bucket_and_object_from_uri(uri: str) -> Tuple[str, str]:
    """Parse a path to determine bucket and object path.

    Args:
        uri: Full path to an object on an object store

    Returns:
        Tuple of strings (bucket_name, object_path)
    """
    if not is_datastore_path(uri):
        raise ValueError(f'Provided URI is not a valid store path: {uri}')
    uri_parts = pathlib.PurePath(uri).parts
    bucket = uri_parts[1]
    object_path = pathlib.PurePath(*uri_parts[2:])

    return str(bucket), str(object_path)


def ais_endpoint_to_dir(endpoint: str) -> str:
    """Convert AIS endpoint to a valid dir name.
    Used to build cache location.

    Args:
        endpoint: AIStore endpoint in format https://host:port
    
    Returns:
        Directory formed as `host/port`.
    """
    if not endpoint.startswith('http://'):
        raise ValueError(f'Unexpected format for ais endpoint: {endpoint}')

    endpoint = endpoint.replace('http://', '')
    host, port = endpoint.split(':')
    return os.path.join(host, port)


def ais_binary() -> str:
    """Return location of `ais` binary.
    """
    path = shutil.which('ais')

    if path is not None:
        logging.debug('Found AIS binary at %s', path)
        return path

    logging.warning('AIS binary not found with `which ais`.')

    # Double-check if it exists at the default path
    default_path = '/usr/local/bin/ais'
    if os.path.isfile(default_path):
        logging.info('ais available at the default path: %s', default_path)
        return default_path
    else:
        raise RuntimeError(f'AIS binary not found.')


def datastore_path_to_local_path(store_path: str) -> str:
    """Convert a data store path to a path in a local cache.

    Args:
        store_path: a path to an object on an object store

    Returns:
        Path to the same object in local cache.
    """
    if store_path.startswith('ais://'):
        endpoint = ais_endpoint()
        if endpoint is None:
            raise RuntimeError(f'AIS endpoint not set, cannot resolve {store_path}')

        local_ais_cache = os.path.join(ais_cache_base(), ais_endpoint_to_dir(endpoint))
        store_bucket, store_object = bucket_and_object_from_uri(store_path)
        local_path = os.path.join(local_ais_cache, store_bucket, store_object)
    else:
        raise ValueError(f'Unexpected store path format: {store_path}')

    return local_path


def get_datastore_object(path: str, force: bool = False, num_retries: int = 5) -> str:
    """Download an object from a store path and return the local path.
    If the input `path` is a local path, then nothing will be done, and
    the original path will be returned.

    Args:
        path: path to an object
        force: force download, even if a local file exists
        num_retries: number of retries if the get command fails
    
    Returns:
        Local path of the object.
    """
    if path.startswith('ais://'):
        endpoint = ais_endpoint()
        if endpoint is None:
            raise RuntimeError(f'AIS endpoint not set, cannot resolve {path}')

        local_path = datastore_path_to_local_path(store_path=path)

        if not os.path.isfile(local_path) or force:
            # Either we don't have the file in cache or we force download it
            # Enhancement: if local file is present, check some tag and compare against remote
            local_dir = os.path.dirname(local_path)
            if not os.path.isdir(local_dir):
                os.makedirs(local_dir, exist_ok=True)

            cmd = [ais_binary(), 'get', path, local_path]

            # for now info, later debug
            logging.debug('Downloading from AIS')
            logging.debug('\tendpoint    %s', endpoint)
            logging.debug('\tpath:       %s', path)
            logging.debug('\tlocal path: %s', local_path)
            logging.debug('\tcmd:        %s', subprocess.list2cmdline(cmd))

            done = False
            for n in range(num_retries):
                if not done:
                    try:
                        # Use stdout=subprocess.DEVNULL to prevent showing AIS command on each line
                        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
                        done = True
                    except subprocess.CalledProcessError as err:
                        logging.warning('Attempt %d of %d failed with: %s', n + 1, num_retries, str(err))

            if not done:
                raise RuntimeError('Download failed: %s', subprocess.list2cmdline(cmd))

        return local_path

    else:
        # Assume the file is local
        return path


class DataStoreObject:
    """A simple class for handling objects in a data store.
    Currently, this class supports objects on AIStore.

    Args:
        store_path: path to a store object
        local_path: path to a local object, may be used to upload local object to store
        get: get the object from a store
    """

    def __init__(self, store_path: str, local_path: str = None, get: bool = False):
        if local_path is not None:
            raise NotImplementedError('Specifying a local path is currently not supported.')

        self._store_path = store_path
        self._local_path = local_path

        if get:
            self.get()

    @property
    def store_path(self) -> str:
        """Return store path of the object.
        """
        return self._store_path

    @property
    def local_path(self) -> str:
        """Return local path of the object.
        """
        return self._local_path

    def get(self, force: bool = False) -> str:
        """Get an object from the store to local cache and return the local path.

        Args:
            force: force download, even if a local file exists

        Returns:
            Path to a local object.
        """
        if not self.local_path:
            # Assume the object needs to be downloaded
            self._local_path = get_datastore_object(self.store_path, force=force)
        return self.local_path

    def put(self, force: bool = False) -> str:
        """Push to remote and return the store path

        Args:
            force: force download, even if a local file exists

        Returns:
            Path to a (remote) object object on the object store.
        """
        raise NotImplementedError()

    def __str__(self):
        """Return a human-readable description of the object.
        """
        description = f'{type(self)}: store_path={self.store_path}, local_path={self.local_path}'
        return description


def datastore_path_to_webdataset_url(store_path: str):
    """Convert store_path to a WebDataset URL.

    Args:
        store_path: path to buckets on store

    Returns:
        URL which can be directly used with WebDataset.
    """
    if store_path.startswith('ais://'):
        url = f'pipe:ais get {store_path} - || true'
    else:
        raise ValueError(f'Unknown store path format: {store_path}')

    return url


def datastore_object_get(store_object: DataStoreObject) -> bool:
    """A convenience wrapper for multiprocessing.imap.

    Args:
        store_object: An instance of DataStoreObject

    Returns:
        True if get() returned a path.
    """
    return store_object.get() is not None
