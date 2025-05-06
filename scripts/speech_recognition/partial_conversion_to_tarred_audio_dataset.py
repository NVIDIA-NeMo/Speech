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

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import hydra
from convert_to_tarred_audio_dataset import ASRTarredDatasetBuilder, ASRTarredDatasetMetadata
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from omegaconf import MISSING
from tqdm import tqdm

"""
# Partial Tarred Audio Dataset Creator

## Overview

This script facilitates the creation of tarred and sharded audio datasets from existing tarred manifests. It allows you to select specific shards from a manifest file and then tar them separately. 

This is useful in several scenarios:
- When you only need to process a specific subset of shards (e.g., for debugging or incremental dataset preparation).
- When you want to parallelize shard creation across multiple SLURM jobs to accelerate the dataset generation process and overcome per-job time limits.

## Prerequisites

- Ensure that the `convert_to_tarred_audio_dataset` script is correctly configured and run with the `--only_manifests` flag to generate the necessary manifest files.
- Make sure the paths to the manifest and metadata files are correct and accessible.

## Usage

### Script Execution

To run the script, use the following command:

python partial_convertion_to_tarred_audio_dataset.py \
    # the path to the tarred manifest file that contains the entries for the shards you want to process. This option is mandatory.
    --tarred_manifest_filepath=<path to the tarred manifest file > \
    # any other optional argument
    --output_dir=<output directory for tarred shards> \
    --shards_to_tar=<shard IDs to be tarred> \
    --num_workers=-1 \
    --dataset_metadata_filepath=<dataset metadata YAML filepath>

Example:
python partial_convertion_to_tarred_audio_dataset.py \
    tarred_manifest_filepath="path/to/manifest.json" \
    shards_to_tar="0:3"
"""


def select_shards(manifest_filepath: str, shards_to_tar: str, slice_with_offset: bool = False):
    """
    Selects and returns a subset of shards from the tarred manifest file.

    Args:
        manifest_filepath (str): The path to the tarred manifest file.
        shards_to_tar (str): A range or list of shard IDs to select, e.g., "0:5" or "0,1,2".
        slice_with_offset (bool, optional): If True, slices entries based on audio offsets. Defaults to False.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        KeyError: If `slice_with_offset` is enabled but required fields are missing in the manifest entries.

    Returns:
        Dict[int, List[Dict[str, any]]]: A dictionary where the keys are shard IDs and the values are lists of entries for those shards.
    """
    shard_ids = []
    if shards_to_tar != "all":
        if ":" not in shards_to_tar:
            shard_ids = [int(shards_to_tar)]
        else:
            start_shard_idx, end_shard_idx = map(
                lambda x: int(x.strip()) if x.strip() else None, shards_to_tar.split(":")
            )
            shard_ids = list(range(start_shard_idx, end_shard_idx))

    entries_to_shard = {}
    with open(manifest_filepath, 'r') as manifest:
        for line in tqdm(manifest, desc="Selecting shards"):
            entry = json.loads(line)
            if shards_to_tar == "all" or entry['shard_id'] in shard_ids:
                if entry['shard_id'] not in entries_to_shard:
                    entries_to_shard[entry['shard_id']] = []

                if slice_with_offset:
                    if 'abs_audio_filepath' not in entry or 'source_audio_offset' not in entry:
                        raise KeyError(
                            f"`slice_with_offset` is enabled, but `abs_audio_filepath` and/or `source_audio_offset` are not found in the entry:\n{entry}."
                        )
                    entry['audio_filepath'] = entry.pop('abs_audio_filepath')
                    entry['offset'] = entry.pop('source_audio_offset')

                entries_to_shard[entry['shard_id']].append(entry)

    return entries_to_shard


@dataclass
class PartialASRTarredDatasetConfig:
    """
    Configuration class for creating partial tarred audio dataset shards.

    Attributes:
        tarred_manifest_filepath (str): The path to the tarred manifest file.
        output_dir (Optional[str]): Directory where the output tarred shards will be saved.
        shards_to_tar (Optional[str]): A range or list of shard IDs to tar.
        num_workers (int): Number of parallel workers to use for tar file creation.
        dataset_metadata_filepath (Optional[str]): Path to the dataset metadata YAML file.
        dataset_metadata (ASRTarredDatasetMetadata): Dataset metadata configuration.
    """

    tarred_manifest_filepath: str = MISSING
    output_dir: Optional[str] = None
    shards_to_tar: Optional[str] = "all"
    num_workers: int = 1
    dataset_metadata_filepath: Optional[str] = None
    dataset_metadata: ASRTarredDatasetMetadata = field(default=ASRTarredDatasetMetadata)
    slice_with_offset: bool = False


def create_shards(cfg: PartialASRTarredDatasetConfig):
    """
    Creates tarred shards based on the provided configuration.

    Args:
        cfg (PartialASRTarredDatasetConfig): The configuration object containing paths, shard IDs, and metadata.

    Raises:
        ValueError: If the `tarred_manifest_filepath` is None.
        FileNotFoundError: If the tarred manifest file or dataset metadata file does not exist.

    Notes:
        - Reads the tarred manifest file and selects the specified shards.
        - Creates tarred shards in parallel using the `ASRTarredDatasetBuilder`.
        - The `dataset_metadata_filepath` is inferred if not provided.
    """
    if cfg.tarred_manifest_filepath is None:
        raise ValueError("The `tarred_manifest_filepath` cannot be `None`. Please check your configuration.")

    if not os.path.exists(cfg.tarred_manifest_filepath):
        raise FileNotFoundError(
            f"The `tarred_manifest_filepath` was not found: {cfg.tarred_manifest_filepath}. Please verify that the filepath is correct."
        )

    if cfg.dataset_metadata_filepath is None:
        cfg.dataset_metadata_filepath = os.path.join(os.path.dirname(cfg.tarred_manifest_filepath), "metadata.yaml")

    if cfg.output_dir is None:
        cfg.output_dir = os.path.dirname(cfg.tarred_manifest_filepath)

    if not os.path.exists(cfg.dataset_metadata_filepath):
        raise FileNotFoundError(
            f"The `dataset_metadata_filepath` was not found: {cfg.dataset_metadata_filepath}. Please verify that the filepath is correct."
        )
    else:
        cfg.dataset_metadata = ASRTarredDatasetMetadata.from_file(cfg.dataset_metadata_filepath)

    entries_to_shard = select_shards(
        cfg.tarred_manifest_filepath, cfg.shards_to_tar, cfg.dataset_metadata.dataset_config.slice_with_offset
    )

    builder = ASRTarredDatasetBuilder()
    builder.configure(cfg.dataset_metadata.dataset_config)

    with Parallel(n_jobs=cfg.num_workers, verbose=len(entries_to_shard)) as parallel:
        # Call parallel tarfile construction
        _ = parallel(
            delayed(builder._create_shard)(
                entries=entries_to_shard[shard_id],
                target_dir=cfg.output_dir,
                shard_id=shard_id,
            )
            for shard_id in entries_to_shard
        )


@hydra.main(config_path=None, config_name='partial_tar_config')
def main(cfg: PartialASRTarredDatasetConfig):
    create_shards(cfg)


ConfigStore.instance().store(name='partial_tar_config', node=PartialASRTarredDatasetConfig)

if __name__ == '__main__':
    main()
