# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import hashlib
import logging
import os
import os.path
import shutil
import tarfile
import tempfile
import urllib.request
from os import mkdir
from os.path import dirname, exists, join
from pathlib import Path
from shutil import rmtree
from typing import Tuple

import pytest

from nemo.utils.metaclasses import Singleton
from nemo.utils.tar_utils import safe_extract

# Those variables probably should go to main NeMo configuration file (config.yaml).
__TEST_DATA_FILENAME = "test_data.tar.gz"
__TEST_DATA_URL = "https://github.com/NVIDIA-NeMo/Speech/releases/download/v1.0.0rc1/"
__TEST_DATA_SUBDIR = ".data"
__TEST_DATA_SHA256 = "bcaf346953ddb7dbd73c67925230886546cfea7120fdab8b199f938ed5960c5d"
__TEST_DATA_MARKER = ".test_data_sha256"


def _get_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _is_test_data_ready(test_dir, test_data_archive):
    marker_path = join(test_dir, __TEST_DATA_MARKER)
    try:
        if _get_sha256(test_data_archive) != __TEST_DATA_SHA256:
            return False
        with open(marker_path, encoding="utf-8") as marker_file:
            return marker_file.read().strip() == __TEST_DATA_SHA256
    except OSError:
        return False


def _mark_test_data_ready(test_dir):
    marker_path = join(test_dir, __TEST_DATA_MARKER)
    with open(marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write(f"{__TEST_DATA_SHA256}\n")


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    For now:
        --cpu: use CPU during testing (DEFAULT: GPU)
        --use_local_test_data: use local test data/skip downloading from URL/GitHub (DEFAULT: False)
    """
    parser.addoption(
        '--cpu', action='store_true', help="pass that argument to use CPU during testing (DEFAULT: False = GPU)"
    )
    parser.addoption(
        '--use_local_test_data',
        action='store_true',
        help="pass that argument to use local test data/skip downloading from URL/GitHub (DEFAULT: False)",
    )
    parser.addoption(
        '--with_downloads',
        action='store_true',
        help="pass this argument to active tests which download models from the cloud.",
    )
    parser.addoption(
        '--relax_numba_compat',
        action='store_false',
        help="numba compatibility checks will be relaxed to just availability of cuda, "
        "without cuda compatibility matrix check",
    )
    parser.addoption(
        "--nightly",
        action="store_true",
        help="pass this argument to activate tests which have been marked as nightly for nightly quality assurance.",
    )


@pytest.fixture
def device(request):
    """Simple fixture returning string denoting the device [CPU | GPU]"""
    if request.config.getoption("--cpu"):
        return "CPU"
    else:
        return "GPU"


@pytest.fixture(autouse=True)
def run_only_on_device_fixture(request, device):
    if request.node.get_closest_marker('run_only_on'):
        if request.node.get_closest_marker('run_only_on').args[0] != device:
            pytest.skip('skipped on this device: {}'.format(device))


@pytest.fixture(autouse=True)
def downloads_weights(request, device):
    if request.node.get_closest_marker('with_downloads'):
        if not request.config.getoption("--with_downloads"):
            pytest.skip(
                'To run this test, pass --with_downloads option. It will download (and cache) models from cloud.'
            )


@pytest.fixture(autouse=True)
def run_nightly_test_for_qa(request, device):
    if request.node.get_closest_marker('nightly'):
        if not request.config.getoption("--nightly"):
            pytest.skip(
                'To run this test, pass --nightly option. It will run any tests marked with "nightly". Currently, These tests are mostly used for QA.'
            )


@pytest.fixture(autouse=True)
def cleanup_local_folder():
    # Asserts in fixture are not recommended, but I'd rather stop users from deleting expensive training runs
    assert not Path("./lightning_logs").exists()
    assert not Path("./NeMo_experiments").exists()
    assert not Path("./nemo_experiments").exists()

    yield

    if Path("./lightning_logs").exists():
        rmtree('./lightning_logs', ignore_errors=True)
    if Path("./NeMo_experiments").exists():
        rmtree('./NeMo_experiments', ignore_errors=True)
    if Path("./nemo_experiments").exists():
        rmtree('./nemo_experiments', ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_singletons():
    Singleton._Singleton__instances = {}


@pytest.fixture(autouse=True)
def reset_env_vars():
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Fixture returns test_data_dir.
    Use the highest fixture scope `session` to allow other fixtures with any other scope to use it.
    """
    # Test dir.
    test_data_dir_ = join(dirname(__file__), __TEST_DATA_SUBDIR)
    return test_data_dir_


def extract_data_from_tar(test_dir, test_data_archive, url=None, local_data=False):
    with tempfile.TemporaryDirectory() as temp_dir:
        staged_archive = join(temp_dir, __TEST_DATA_FILENAME)
        if url is not None and not local_data:
            urllib.request.urlretrieve(url, staged_archive)
        else:
            shutil.copy2(test_data_archive, staged_archive)

        if _get_sha256(staged_archive) != __TEST_DATA_SHA256:
            raise ValueError("Test data does not match the pinned SHA-256 checksum")

        if exists(test_dir):
            rmtree(test_dir)
        mkdir(test_dir)
        shutil.copy2(staged_archive, test_data_archive)

    print("Extracting the `{}` test archive, please wait...".format(test_data_archive))
    with tarfile.open(test_data_archive) as tar:
        safe_extract(tar, test_dir)
    _mark_test_data_ready(test_dir)


@pytest.fixture(scope="session")
def k2_is_appropriate() -> Tuple[bool, str]:
    try:
        from nemo.core.utils.k2_guard import k2  # noqa: E402

        return True, "k2 is appropriate."
    except Exception as e:
        logging.exception(e, exc_info=True)
        return False, "k2 is not available or does not meet the requirements."


@pytest.fixture(scope="session")
def k2_cuda_is_enabled(k2_is_appropriate) -> Tuple[bool, str]:
    if not k2_is_appropriate[0]:
        return k2_is_appropriate

    import torch  # noqa: E402

    from nemo.core.utils.k2_guard import k2  # noqa: E402

    if torch.cuda.is_available() and k2.with_cuda:
        return True, "k2 supports CUDA."
    elif torch.cuda.is_available():
        return False, "k2 does not support CUDA. Consider using a k2 build with CUDA support."
    else:
        return False, "k2 needs CUDA to be available in torch."


def pytest_configure(config):
    """
    Initial configuration of conftest.
    The function accepts checksum-validated, extracted test data in tests/.data.
    Otherwise, it downloads, validates, and extracts the pinned archive from GitHub.
    """
    config.addinivalue_line(
        "markers",
        "run_only_on(device): runs the test only on a given device [CPU | GPU]",
    )
    config.addinivalue_line(
        "markers",
        "with_downloads: runs the test using data present in tests/.data",
    )
    config.addinivalue_line(
        "markers",
        "nightly: runs the nightly test for QA.",
    )
    # Test dir and archive filepath.
    test_dir = join(dirname(__file__), __TEST_DATA_SUBDIR)
    test_data_archive = join(dirname(__file__), __TEST_DATA_SUBDIR, __TEST_DATA_FILENAME)

    if _is_test_data_ready(test_dir, test_data_archive):
        print("Using checksum-validated test data found in the `{}` folder.".format(test_dir))
        return

    if exists(test_data_archive) and _get_sha256(test_data_archive) == __TEST_DATA_SHA256:
        extract_data_from_tar(test_dir, test_data_archive, local_data=True)
        return

    if config.option.use_local_test_data:
        pytest.exit("Test data not present or invalid in the `{}` folder".format(test_dir))

    url = __TEST_DATA_URL + __TEST_DATA_FILENAME
    try:
        extract_data_from_tar(test_dir, test_data_archive, url=url)
    except (OSError, ValueError, tarfile.TarError):
        pytest.exit("Test data is unavailable or invalid at '{}'".format(url))
