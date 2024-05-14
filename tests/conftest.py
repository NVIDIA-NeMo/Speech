# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os.path
import shutil
import tarfile
import tempfile
import urllib.request
from os import mkdir
from os.path import dirname, exists, getsize, join
from pathlib import Path
from shutil import rmtree
from typing import Tuple

import pytest

# Those variables probably should go to main NeMo configuration file (config.yaml).
__TEST_DATA_FILENAME = "test_data.tar.gz"
__TEST_DATA_URL = "https://github.com/NVIDIA/NeMo/releases/download/v1.0.0rc1/"
__TEST_DATA_SUBDIR = ".data"


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
    # Remove .data folder.
    if exists(test_dir):
        if not local_data:
            rmtree(test_dir)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                print("Copying local tarfile to temporary storage..")
                shutil.copy2(test_data_archive, temp_dir)
                print("Deleting test dir to cleanup old data")
                rmtree(test_dir)
                mkdir(test_dir)
                print("Restoring local tarfile to test dir")
                shutil.copy2(os.path.join(temp_dir, os.path.basename(test_data_archive)), test_data_archive)

    # Create one .data folder.
    if not exists(test_dir):
        mkdir(test_dir)

    # Download (if required)
    if url is not None and not local_data:
        urllib.request.urlretrieve(url, test_data_archive)

    # Extract tar
    print("Extracting the `{}` test archive, please wait...".format(test_data_archive))
    tar = tarfile.open(test_data_archive)
    tar.extractall(path=test_dir)
    tar.close()


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
    The function checks if test_data.tar.gz is present in tests/.data.
    If so, compares its size with github's test_data.tar.gz.
    If file absent or sizes not equal, function downloads the archive from github and unpacks it.
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

    # Get size of local test_data archive.
    try:
        test_data_local_size = getsize(test_data_archive)
    except:
        # File does not exist.
        test_data_local_size = -1

    if config.option.use_local_test_data:
        if test_data_local_size == -1:
            pytest.exit("Test data `{}` is not present in the system".format(test_data_archive))
        else:
            print(
                "Using the local `{}` test archive ({}B) found in the `{}` folder.".format(
                    __TEST_DATA_FILENAME, test_data_local_size, test_dir
                )
            )

    # Get size of remote test_data archive.
    url = None
    if not config.option.use_local_test_data:
        try:
            url = __TEST_DATA_URL + __TEST_DATA_FILENAME
            u = urllib.request.urlopen(url)
        except:
            # Couldn't access remote archive.
            if test_data_local_size == -1:
                pytest.exit("Test data not present in the system and cannot access the '{}' URL".format(url))
            else:
                print(
                    "Cannot access the '{}' URL, using the test data ({}B) found in the `{}` folder.".format(
                        url, test_data_local_size, test_dir
                    )
                )
                return

        # Get metadata.
        meta = u.info()
        test_data_remote_size = int(meta["Content-Length"])

        # Compare sizes.
        if test_data_local_size != test_data_remote_size:
            print(
                "Downloading the `{}` test archive from `{}`, please wait...".format(
                    __TEST_DATA_FILENAME, __TEST_DATA_URL
                )
            )

            extract_data_from_tar(test_dir, test_data_archive, url=url, local_data=config.option.use_local_test_data)

        else:
            print(
                "A valid `{}` test archive ({}B) found in the `{}` folder.".format(
                    __TEST_DATA_FILENAME, test_data_local_size, test_dir
                )
            )

    else:
        # untar local test data
        extract_data_from_tar(test_dir, test_data_archive, local_data=config.option.use_local_test_data)

    if config.option.relax_numba_compat is not None:
        from nemo.core.utils import numba_utils

        print("Setting numba compat :", config.option.relax_numba_compat)
        numba_utils.set_numba_compat_strictness(strict=config.option.relax_numba_compat)
