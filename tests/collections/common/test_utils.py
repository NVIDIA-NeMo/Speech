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
import string
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from nemo.collections.common.parts.preprocessing.manifest import get_full_path, is_tarred_dataset
from nemo.collections.common.parts.utils import flatten, mask_sequence_tensor


class TestListUtils:
    @pytest.mark.unit
    def test_flatten(self):
        """Test flattening an iterable with different values: str, bool, int, float, complex."""
        test_cases = []
        test_cases.append({'input': ['aa', 'bb', 'cc'], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', ['bb', 'cc']], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', [['bb'], [['cc']]]], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', [[1, 2], [[3]], 4]], 'golden': ['aa', 1, 2, 3, 4]})
        test_cases.append({'input': [True, [2.5, 2.0 + 1j]], 'golden': [True, 2.5, 2.0 + 1j]})

        for n, test_case in enumerate(test_cases):
            assert flatten(test_case['input']) == test_case['golden'], f'Test case {n} failed!'


class TestMaskSequenceTensor:
    @pytest.mark.unit
    @pytest.mark.parametrize('ndim', [2, 3, 4, 5])
    def test_mask_sequence_tensor(self, ndim: int):
        """Test masking a tensor based on the provided length."""
        num_examples = 20
        max_batch_size = 10
        max_max_len = 30

        for n in range(num_examples):
            batch_size = np.random.randint(low=1, high=max_batch_size)
            max_len = np.random.randint(low=1, high=max_max_len)

            if ndim > 2:
                tensor_shape = (batch_size,) + tuple(torch.randint(1, 30, (ndim - 2,))) + (max_len,)
            else:
                tensor_shape = (batch_size, max_len)

            tensor = torch.randn(tensor_shape)
            lengths = torch.randint(low=1, high=max_len + 1, size=(batch_size,))

            if ndim <= 4:
                masked_tensor = mask_sequence_tensor(tensor=tensor, lengths=lengths)

                for b, l in enumerate(lengths):
                    assert torch.equal(masked_tensor[b, ..., :l], tensor[b, ..., :l]), f'Failed for example {n}'
                    assert torch.all(masked_tensor[b, ..., l:] == 0.0), f'Failed for example {n}'
            else:
                # Currently, supporting only up to 4D tensors
                with pytest.raises(ValueError):
                    mask_sequence_tensor(tensor=tensor, lengths=lengths)


class TestPreprocessingUtils:
    @pytest.mark.unit
    def test_get_full_path_local(self, tmpdir):
        """Test with local paths"""
        # Create a few files
        num_files = 10

        audio_files_relative_path = [f'file_{n}.test' for n in range(num_files)]
        audio_files_absolute_path = [os.path.join(tmpdir, a_file_rel) for a_file_rel in audio_files_relative_path]

        data_dir = tmpdir
        manifest_file = os.path.join(data_dir, 'manifest.json')

        # Context manager to create dummy files
        @contextmanager
        def create_files(paths):
            # Create files
            for a_file in paths:
                Path(a_file).touch()
            yield
            # Remove files
            for a_file in paths:
                Path(a_file).unlink()

        # 1) Test with absolute paths and while files don't exist.
        # Note: it's still expected the path will be resolved correctly, since it will be
        # expanded using manifest_file.parent or data_dir and relative path.
        # - single file
        for n in range(num_files):
            assert (
                get_full_path(audio_files_absolute_path[n], manifest_file=manifest_file)
                == audio_files_absolute_path[n]
            )
            assert get_full_path(audio_files_absolute_path[n], data_dir=data_dir) == audio_files_absolute_path[n]

        # - all files in a list
        assert get_full_path(audio_files_absolute_path, manifest_file=manifest_file) == audio_files_absolute_path
        assert get_full_path(audio_files_absolute_path, data_dir=data_dir) == audio_files_absolute_path

        # 2) Test with absolute paths and existing files.
        with create_files(audio_files_absolute_path):
            # - single file
            for n in range(num_files):
                assert (
                    get_full_path(audio_files_absolute_path[n], manifest_file=manifest_file)
                    == audio_files_absolute_path[n]
                )
                assert get_full_path(audio_files_absolute_path[n], data_dir=data_dir) == audio_files_absolute_path[n]

            # - all files in a list
            assert get_full_path(audio_files_absolute_path, manifest_file=manifest_file) == audio_files_absolute_path
            assert get_full_path(audio_files_absolute_path, data_dir=data_dir) == audio_files_absolute_path

        # 3) Test with relative paths while files don't exist.
        # This is a situation we may have with a tarred dataset.
        # In this case, we expect to return the relative path.
        # - single file
        for n in range(num_files):
            assert (
                get_full_path(audio_files_relative_path[n], manifest_file=manifest_file)
                == audio_files_relative_path[n]
            )
            assert get_full_path(audio_files_relative_path[n], data_dir=data_dir) == audio_files_relative_path[n]

        # - all files in a list
        assert get_full_path(audio_files_relative_path, manifest_file=manifest_file) == audio_files_relative_path
        assert get_full_path(audio_files_relative_path, data_dir=data_dir) == audio_files_relative_path

        # 4) Test with relative paths and existing files.
        # In this case, we expect to return the absolute path.
        with create_files(audio_files_absolute_path):
            # - single file
            for n in range(num_files):
                assert (
                    get_full_path(audio_files_relative_path[n], manifest_file=manifest_file)
                    == audio_files_absolute_path[n]
                )
                assert get_full_path(audio_files_relative_path[n], data_dir=data_dir) == audio_files_absolute_path[n]

            # - all files in a list
            assert get_full_path(audio_files_relative_path, manifest_file=manifest_file) == audio_files_absolute_path
            assert get_full_path(audio_files_relative_path, data_dir=data_dir) == audio_files_absolute_path

        # 5) Test with relative paths and existing files, and the filepaths start with './'.
        # In this case, we expect to return the same relative path.
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        audio_files_relative_path_curr = [f'./file_{n}.test' for n in range(num_files)]
        with create_files(audio_files_relative_path_curr):
            # - single file
            for n in range(num_files):
                assert os.path.isfile(audio_files_relative_path_curr[n]) == True
                assert (
                    get_full_path(audio_files_relative_path_curr[n], manifest_file=manifest_file)
                    == audio_files_relative_path_curr[n]
                )
                assert (
                    get_full_path(audio_files_relative_path_curr[n], data_dir=curr_dir)
                    == audio_files_relative_path_curr[n]
                )

            # - all files in a list
            assert (
                get_full_path(audio_files_relative_path_curr, manifest_file=manifest_file)
                == audio_files_relative_path_curr
            )
            assert get_full_path(audio_files_relative_path_curr, data_dir=curr_dir) == audio_files_relative_path_curr

    @pytest.mark.unit
    def test_get_full_path_ais(self, tmpdir):
        """Test with paths on AIStore."""
        # Create a few files
        num_files = 10

        audio_files_relative_path = [f'file_{n}.test' for n in range(num_files)]
        audio_files_cache_path = [os.path.join(tmpdir, a_file_rel) for a_file_rel in audio_files_relative_path]

        ais_data_dir = 'ais://test'
        ais_manifest_file = os.path.join(ais_data_dir, 'manifest.json')

        # Context manager to create dummy files
        @contextmanager
        def create_files(paths):
            # Create files
            for a_file in paths:
                Path(a_file).touch()
            yield
            # Remove files
            for a_file in paths:
                Path(a_file).unlink()

        # Simulate caching in local tmpdir
        def datastore_path_to_cache_path_in_tmpdir(path):
            rel_path = os.path.relpath(path, start=os.path.dirname(ais_manifest_file))

            if rel_path in audio_files_relative_path:
                idx = audio_files_relative_path.index(rel_path)
                return audio_files_cache_path[idx]
            else:
                raise ValueError(f'Unexpected path {path}')

        with mock.patch(
            'nemo.collections.common.parts.preprocessing.manifest.datastore_path_to_local_path',
            datastore_path_to_cache_path_in_tmpdir,
        ):
            # Test with relative paths and existing cached files.
            # We expect to return the absolute path in the local cache.
            with create_files(audio_files_cache_path):
                # - single file
                for n in range(num_files):
                    assert (
                        get_full_path(audio_files_relative_path[n], manifest_file=ais_manifest_file)
                        == audio_files_cache_path[n]
                    )
                    assert (
                        get_full_path(audio_files_relative_path[n], data_dir=ais_data_dir) == audio_files_cache_path[n]
                    )

                # - all files in a list
                assert (
                    get_full_path(audio_files_relative_path, manifest_file=ais_manifest_file) == audio_files_cache_path
                )
                assert get_full_path(audio_files_relative_path, data_dir=ais_data_dir) == audio_files_cache_path

    @pytest.mark.unit
    def test_get_full_path_audio_file_len_limit(self):
        """Test with audio_file_len_limit.
        Currently, get_full_path will always return the input path when the length
        is over audio_file_len_limit, independend of whether the file exists.
        """
        # Create a few files
        num_examples = 10
        rand_chars = list(string.ascii_uppercase + string.ascii_lowercase + string.digits + os.sep)
        rand_name = lambda n: ''.join(np.random.choice(rand_chars, size=n))

        for audio_file_len_limit in [255, 300]:
            for n in range(num_examples):
                path_length = np.random.randint(low=audio_file_len_limit, high=350)
                audio_file_path = str(Path(rand_name(path_length)))

                assert (
                    get_full_path(audio_file_path, audio_file_len_limit=audio_file_len_limit) == audio_file_path
                ), f'Limit {audio_file_len_limit}: expected {audio_file_path} to be returned.'

                audio_file_path_with_user = os.path.join('~', audio_file_path)
                audio_file_path_with_user_expected = os.path.expanduser(audio_file_path_with_user)
                assert (
                    get_full_path(audio_file_path_with_user, audio_file_len_limit=audio_file_len_limit)
                    == audio_file_path_with_user_expected
                ), f'Limit {audio_file_len_limit}: expected {audio_file_path_with_user_expected} to be returned.'

    @pytest.mark.unit
    def test_get_full_path_invalid_type(self):
        """Make sure exceptions are raised when audio_file is not a string or a list of strings."""

        with pytest.raises(ValueError, match="Unexpected audio_file type"):
            get_full_path(1)

        with pytest.raises(ValueError, match="Unexpected audio_file type"):
            get_full_path(('a', 'b', 'c'))

        with pytest.raises(ValueError, match="Unexpected audio_file type"):
            get_full_path({'a': 1, 'b': 2, 'c': 3})

        with pytest.raises(ValueError, match="Unexpected audio_file type"):
            get_full_path([1, 2, 3])

    @pytest.mark.unit
    def test_get_full_path_invalid_relative_path(self):
        """Make sure exceptions are raised when audio_file is a relative path and
        manifest is not provided or both manifest and data dir are provided simultaneously.
        """
        with pytest.raises(ValueError, match="Use either manifest_file or data_dir"):
            # Using a relative path without manifest_file or data_dir is not allowed
            get_full_path('relative/path')

        with pytest.raises(ValueError, match="Parameters manifest_file and data_dir cannot be used simultaneously."):
            # Using a relative path without both manifest_file or data_dir is not allowed
            get_full_path('relative/path', manifest_file='/manifest_dir/file.json', data_dir='/data/dir')

    @pytest.mark.unit
    def test_is_tarred_dataset(self):
        # 1) is tarred dataset
        assert is_tarred_dataset("_file_1.wav", "tarred_audio_manifest.json")
        assert is_tarred_dataset("_file_1.wav", "./sharded_manifests/manifest_1.json")

        # 2) is not tarred dataset
        assert not is_tarred_dataset("./file_1.wav", "audio_manifest.json")
        assert not is_tarred_dataset("./file_1.wav", "./sharded_manifests/manifest_test.json")
        assert not is_tarred_dataset("file_1.wav", "audio_manifest.json")
        assert not is_tarred_dataset("file_1.wav", "./sharded_manifests/manifest_test.json")
        assert not is_tarred_dataset("/data/file_1.wav", "audio_manifest.json")
        assert not is_tarred_dataset("/data/file_1.wav", "./sharded_manifests/manifest_test.json")
        assert not is_tarred_dataset("_file_1.wav", "audio_manifest.json")
        assert not is_tarred_dataset("_file_1.wav", "./sharded_manifests/manifest_test.json")

        # 3) no manifest file, treated as non-tarred dataset
        assert not is_tarred_dataset("_file_1.wav", None)
