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

import os
from pathlib import Path
from time import sleep

import wget
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import DDPStrategy, StrategyRegistry

from nemo.utils import logging


def maybe_download_from_cloud(url, filename, subfolder=None, cache_dir=None, refresh_cache=False) -> str:
    """
    Helper function to download pre-trained weights from the cloud
    Args:
        url: (str) URL of storage
        filename: (str) what to download. The request will be issued to url/filename
        subfolder: (str) subfolder within cache_dir. The file will be stored in cache_dir/subfolder. Subfolder can
            be empty
        cache_dir: (str) a cache directory where to download. If not present, this function will attempt to create it.
            If None (default), then it will be $HOME/.cache/torch/NeMo
        refresh_cache: (bool) if True and cached file is present, it will delete it and re-fetch

    Returns:
        If successful - absolute local path to the downloaded file
        else - empty string
    """
    # try:
    if cache_dir is None:
        cache_location = Path.joinpath(Path.home(), ".cache/torch/NeMo")
    else:
        cache_location = cache_dir
    if subfolder is not None:
        destination = Path.joinpath(cache_location, subfolder)
    else:
        destination = cache_location

    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    destination_file = Path.joinpath(destination, filename)

    if os.path.exists(destination_file):
        logging.info(f"Found existing object {destination_file}.")
        if refresh_cache:
            logging.info("Asked to refresh the cache.")
            logging.info(f"Deleting file: {destination_file}")
            os.remove(destination_file)
        else:
            logging.info(f"Re-using file from: {destination_file}")
            return str(destination_file)
    # download file
    wget_uri = url + filename
    logging.info(f"Downloading from: {wget_uri} to {str(destination_file)}")
    # NGC links do not work everytime so we try and wait
    i = 0
    max_attempts = 3
    while i < max_attempts:
        i += 1
        try:
            wget.download(wget_uri, str(destination_file))
            if os.path.exists(destination_file):
                return destination_file
            else:
                return ""
        except:
            logging.info(f"Download from cloud failed. Attempt {i} of {max_attempts}")
            sleep(0.05)
            continue
    raise ValueError("Not able to download url right now, please try again.")


class SageMakerDDPStrategy(DDPStrategy):
    """DDP strategy configured for AWS SageMaker distributed training."""

    @property
    def cluster_environment(self):
        """Return a LightningEnvironment configured from SageMaker environment variables."""
        env = LightningEnvironment()
        env.world_size = lambda: int(os.environ["WORLD_SIZE"])
        env.global_rank = lambda: int(os.environ["RANK"])
        return env

    @cluster_environment.setter
    def cluster_environment(self, env):
        """No-op setter to prevent Lightning from overriding the SageMaker environment."""
        pass


def initialize_sagemaker() -> None:
    """
    Helper function to initiate sagemaker with NeMo.
    This function installs libraries that NeMo requires for the ASR toolkit + initializes sagemaker ddp.
    """

    logging.info("Registering SageMaker DDP strategy 'smddp'.")
    StrategyRegistry.register(
        name='smddp',
        strategy=SageMakerDDPStrategy,
        process_group_backend="smddp",
        find_unused_parameters=False,
    )

    def _install_system_libraries() -> None:
        import subprocess

        logging.info("Installing system libraries: libsndfile1, ffmpeg")
        try:
            logging.info("Running apt-get update")
            subprocess.run(
                ["apt-get", "update"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logging.info("Running apt-get install for libsndfile1 and ffmpeg")
            subprocess.run(
                ["apt-get", "install", "-y", "libsndfile1", "ffmpeg"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            logging.error(
                "Failed to install system libraries via apt-get (command=%s, returncode=%s): %s",
                getattr(e, "cmd", None),
                getattr(e, "returncode", None),
                e,
            )
            logging.info("System libraries installed successfully.")
        except Exception as e:
            logging.error(f"Failed to install system libraries: {e}")

    def _patch_torch_metrics() -> None:
        """
        Patches torchmetrics to not rely on internal state.
        This is because sagemaker DDP overrides the `__init__` function of the modules to do automatic-partitioning.
        """
        from torchmetrics import Metric

        def __new_hash__(self):
            hash_vals = [self.__class__.__name__, id(self)]
            return hash(tuple(hash_vals))

        Metric.__hash__ = __new_hash__

    logging.info("Patching torchmetrics hash function for SageMaker compatibility.")
    _patch_torch_metrics()

    if os.environ.get("RANK") and os.environ.get("WORLD_SIZE"):
        import smdistributed.dataparallel.torch.distributed as dist

        # has to be imported, as it overrides torch modules and such when DDP is enabled.
        import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401

        logging.info("Initializing SageMaker distributed process group.")
        dist.init_process_group()

        if dist.get_local_rank():
            _install_system_libraries()
        logging.info("Waiting at barrier for all processes.")
        return dist.barrier()  # wait for main process
    _install_system_libraries()
    return
