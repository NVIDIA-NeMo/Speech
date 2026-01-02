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
import argparse
import glob
import math
import os
from typing import List, Union


from filelock import FileLock
from omegaconf import ListConfig, OmegaConf


def count_files_for_tarred_pseudo_labeling(manifest_filepath: Union[str, ListConfig]) -> int:
    """
    Counts the total number of entries across multiple manifest files.
    Args:
        manifest_filepath (Union[str, ListConfig]): The file path to the manifest files.
    Returns:
        int: The total number of entries across all matching manifest files.
    """
    # Convert ListConfig to string if needed
    if isinstance(manifest_filepath, ListConfig):
        manifest_filepath = manifest_filepath[0]  # Use the first element if it's a list or ListConfig
    dir_path, filename = os.path.split(manifest_filepath)
    prefix = filename.split('_', 1)[0]
    number_of_files = 0
    for full_path in glob.glob(os.path.join(dir_path, f"{prefix}_[0-9]*.json")):
        with open(full_path, 'r') as f:
            number_of_files += len(f.readlines())
    return number_of_files


def count_files_for_pseudo_labeling(manifest_filepath: Union[str, list, ListConfig]) -> int:
    """
    Counts the number of entries in a single manifest file .
    Args:
        manifest_filepath (Union[str, list, ListConfig]): The file path to the manifest file.
    Returns:
        int: The total number of entries (lines) in the manifest file.
    """
    # Convert ListConfig to string if needed
    if isinstance(manifest_filepath, list) or isinstance(manifest_filepath, ListConfig):
        manifest_filepath = manifest_filepath[0]
    with open(manifest_filepath, 'r') as f:
        number_of_files = len(f.readlines())
    return number_of_files


def export_limit_predict_batches(inference_configs: List[str], p_cache: float, num_gpus: int) -> None:
    """
    Updates inference configuration files to set `limit_predict_batches`.
    This is done to force partial transcription of unlabeled dataset for dynamic update of PLs.

    Args:
        inference_configs (List[str]): A list of file paths to the inference configuration files.
        p_cache (float): A scaling factor for the cache to adjust the number of batches.
        num_gpus (int): The number of GPUs available for inference.

    Returns:
        None: The function modifies and saves the updated configuration files in-place.
    """
    for config_path in inference_configs:
        config = OmegaConf.load(config_path)
        tarred_audio_filepaths = config.predict_ds.get("tarred_audio_filepaths", None)
        manifest_filepaths = config.predict_ds.manifest_filepath

        if tarred_audio_filepaths:
            number_of_files = count_files_for_tarred_pseudo_labeling(manifest_filepaths)
        else:
            number_of_files = count_files_for_pseudo_labeling(manifest_filepaths)

        if hasattr(config.predict_ds, "batch_size"):
            batch_size = config.predict_ds.batch_size
            limit_predict_batches = math.ceil((number_of_files * p_cache) / (batch_size * num_gpus))
            OmegaConf.update(config, "trainer.limit_predict_batches", limit_predict_batches)
            OmegaConf.save(config, config_path)
        elif hasattr(config.predict_ds, "batch_duration"):
            batch_duration = config.predict_ds.batch_duration
            average_audio_len = 10
            limit_predict_batches = math.ceil(
                (number_of_files * average_audio_len * p_cache) / (batch_duration * num_gpus)
            )
            OmegaConf.update(config, "trainer.limit_predict_batches", limit_predict_batches)
            OmegaConf.save(config, config_path)
        else:
            batch_size = 32
            limit_predict_batches = math.ceil((number_of_files * p_cache) / (batch_size * num_gpus))
            OmegaConf.update(config, "trainer.limit_predict_batches", limit_predict_batches)
            OmegaConf.save(config, config_path)


def main():
    rank = int(os.environ.get("RANK", 0))  # Default to 0 if not set

    # Ensure only one process executes this block
    parser = argparse.ArgumentParser(description="Export limit_predict_batches as environment variables.")
    parser.add_argument(
        "--inference_configs",
        type=str,
        nargs='+',  # Accepts one or more values as a list
        required=True,
        help="Paths to one or more inference config YAML files.",
    )
    parser.add_argument("--p_cache", type=float, required=True, help="Pseudo-label cache fraction.")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs available.")

    args = parser.parse_args()
    lock_dir = os.path.dirname(args.inference_configs[0])
    lock_file = lock_dir + "/my_script.lock"
    # Code executed by all processes

    #     # Code executed by a single process
    with FileLock(lock_file):
        if rank == 0:
            export_limit_predict_batches(
                inference_configs=args.inference_configs, p_cache=args.p_cache, num_gpus=args.num_gpus
            )

    # Remove the lock file after the FileLock context is exited
    if os.path.exists(lock_file):
        os.remove(lock_file)


if __name__ == "__main__":
    main()
