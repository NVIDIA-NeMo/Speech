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
"""
# This script converts an existing audio dataset with a manifest to
# a tarred and sharded audio dataset that can be read by the
# TarredAudioToTextDataLayer.

# Please make sure your audio_filepath DOES NOT CONTAIN '-sub'!
# Because we will use it to handle files which have duplicate filenames but with different offsets
# (see function create_shard for details)


# Bucketing can help to improve the training speed. You may use --buckets_num to specify the number of buckets.
# It creates multiple tarred datasets, one per bucket, based on the audio durations.
# The range of [min_duration, max_duration) is split into equal sized buckets.
# Recommend to use --sort_in_shards to speedup the training by reducing the paddings in the batches
# More info on how to use bucketing feature: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/datasets.html

# If valid NVIDIA DALI version is installed, will also generate the corresponding DALI index files that need to be
# supplied to the config in order to utilize webdataset for efficient large dataset handling.
# NOTE: DALI + Webdataset is NOT compatible with Bucketing support !

# Usage:
1) Creating a new tarfile dataset

python convert_to_tarred_audio_dataset.py \
    --manifest_path=<path to the manifest file> \
    --target_dir=<path to output directory> \
    --num_shards=<number of tarfiles that will contain the audio> \
    --max_duration=<float representing maximum duration of audio samples> \
    --min_duration=<float representing minimum duration of audio samples> \
    --shuffle --shuffle_seed=1 \
    --sort_in_shards \
    --force_codec=flac \
    --workers=-1


2) Concatenating more tarfiles to a pre-existing tarred dataset

python convert_to_tarred_audio_dataset.py \
    --manifest_path=<path to the tarred manifest file> \
    --metadata_path=<path to the metadata.yaml (or metadata_version_{X}.yaml) file> \
    --target_dir=<path to output directory where the original tarfiles are contained> \
    --max_duration=<float representing maximum duration of audio samples> \
    --min_duration=<float representing minimum duration of audio samples> \
    --shuffle --shuffle_seed=1 \
    --sort_in_shards \
    --workers=-1 \
    --concat_manifest_paths
    <space separated paths to 1 or more manifest files to concatenate into the original tarred dataset>

3) Writing an empty metadata file

python convert_to_tarred_audio_dataset.py \
    --target_dir=<path to output directory> \
    # any other optional argument
    --num_shards=8 \
    --max_duration=16.7 \
    --min_duration=0.01 \
    --shuffle \
    --workers=-1 \
    --sort_in_shards \
    --shuffle_seed=1 \
    --write_metadata

"""
import argparse
import copy
import json
import os
import random
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any, List, Optional

import numpy as np
import soundfile
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf, open_dict

try:
    import create_dali_tarred_dataset_index as dali_index

    DALI_INDEX_SCRIPT_AVAILABLE = True
except (ImportError, ModuleNotFoundError, FileNotFoundError):
    DALI_INDEX_SCRIPT_AVAILABLE = False

parser = argparse.ArgumentParser(
    description="Convert an existing ASR dataset to tarballs compatible with TarredAudioToTextDataLayer."
)
parser.add_argument(
    "--manifest_path", default=None, type=str, required=False, help="Path to the existing dataset's manifest."
)

parser.add_argument(
    '--concat_manifest_paths',
    nargs='+',
    default=None,
    type=str,
    required=False,
    help="Path to the additional dataset's manifests that will be concatenated with base dataset.",
)

# Optional arguments
parser.add_argument(
    "--target_dir",
    default='./tarred',
    type=str,
    help="Target directory for resulting tarballs and manifest. Defaults to `./tarred`. Creates the path if necessary.",
)

parser.add_argument(
    "--metadata_path",
    required=False,
    default=None,
    type=str,
    help="Path to metadata file for the dataset.",
)

parser.add_argument(
    "--num_shards",
    default=-1,
    type=int,
    help="Number of shards (tarballs) to create. Used for partitioning data among workers.",
)
parser.add_argument(
    '--max_duration',
    default=None,
    required=True,
    type=float,
    help='Maximum duration of audio clip in the dataset. By default, it is None and is required to be set.',
)
parser.add_argument(
    '--min_duration',
    default=None,
    type=float,
    help='Minimum duration of audio clip in the dataset. By default, it is None and will not filter files.',
)
parser.add_argument(
    "--shuffle",
    action='store_true',
    help="Whether or not to randomly shuffle the samples in the manifest before tarring/sharding.",
)

parser.add_argument(
    "--keep_files_together",
    action='store_true',
    help="Whether or not to keep entries from the same file (but different offsets) together when sorting before tarring/sharding.",
)

parser.add_argument(
    "--sort_in_shards",
    action='store_true',
    help="Whether or not to sort samples inside the shards based on their duration.",
)

parser.add_argument(
    "--buckets_num",
    type=int,
    default=1,
    help="Number of buckets to create based on duration.",
)

parser.add_argument(
    "--dynamic_buckets_num",
    type=int,
    default=30,
    help="Intended for dynamic (on-the-fly) bucketing; this option will not bucket your dataset during tar conversion. "
    "Estimates optimal bucket duration bins for a given number of buckets.",
)

parser.add_argument("--shuffle_seed", type=int, default=None, help="Random seed for use if shuffling is enabled.")
parser.add_argument(
    '--write_metadata',
    action='store_true',
    help=(
        "Flag to write a blank metadata with the current call config. "
        "Note that the metadata will not contain the number of shards, "
        "and it must be filled out by the user."
    ),
)
parser.add_argument(
    "--no_shard_manifests",
    action='store_true',
    help="Do not write sharded manifests along with the aggregated manifest.",
)
parser.add_argument(
    "--force_codec",
    type=str,
    default=None,
    help=(
        "If specified, transcode the audio to the given format. "
        "Supports libnsndfile formats (example values: 'opus', 'flac')."
    ),
)
parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
args = parser.parse_args()


@dataclass
class ASRTarredDatasetConfig:
    num_shards: int = -1
    shuffle: bool = False
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    shuffle_seed: Optional[int] = None
    sort_in_shards: bool = True
    shard_manifests: bool = True
    keep_files_together: bool = False
    force_codec: Optional[str] = None
    use_lhotse: bool = False
    use_bucketing: bool = False
    num_buckets: Optional[int] = None
    bucket_duration_bins: Optional[list[float]] = None


@dataclass
class ASRTarredDatasetMetadata:
    created_datetime: Optional[str] = None
    version: int = 0
    num_samples_per_shard: Optional[int] = None
    is_concatenated_manifest: bool = False

    dataset_config: Optional[ASRTarredDatasetConfig] = field(default_factory=lambda: ASRTarredDatasetConfig())
    history: Optional[List[Any]] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.created_datetime = self.get_current_datetime()

    def get_current_datetime(self):
        return datetime.now().strftime("%m-%d-%Y %H-%M-%S")

    @classmethod
    def from_config(cls, config: DictConfig):
        obj = cls()
        obj.__dict__.update(**config)
        return obj

    @classmethod
    def from_file(cls, filepath: str):
        config = OmegaConf.load(filepath)
        return ASRTarredDatasetMetadata.from_config(config=config)


class ASRTarredDatasetBuilder:
    """
    Helper class that constructs a tarred dataset from scratch, or concatenates tarred datasets
    together and constructs manifests for them.
    """

    def __init__(self):
        self.config = None

    def configure(self, config: ASRTarredDatasetConfig):
        """
        Sets the config generated from command line overrides.

        Args:
            config: ASRTarredDatasetConfig dataclass object.
        """
        self.config = config  # type: ASRTarredDatasetConfig

        if self.config.num_shards < 0:
            raise ValueError("`num_shards` must be > 0. Please fill in the metadata information correctly.")

    def create_new_dataset(self, manifest_path: str, target_dir: str = "./tarred/", num_workers: int = 0):
        """
        Creates a new tarred dataset from a given manifest file.

        Args:
            manifest_path: Path to the original ASR manifest.
            target_dir: Output directory.
            num_workers: Integer denoting number of parallel worker processes which will write tarfiles.
                Defaults to 1 - which denotes sequential worker process.

        Output:
            Writes tarfiles, along with the tarred dataset compatible manifest file.
            Also preserves a record of the metadata used to construct this tarred dataset.
        """
        if self.config is None:
            raise ValueError("Config has not been set. Please call `configure(config: ASRTarredDatasetConfig)`")

        if manifest_path is None:
            raise FileNotFoundError("Manifest filepath cannot be None !")

        config = self.config  # type: ASRTarredDatasetConfig

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Read the existing manifest
        entries, total_duration, filtered_entries, filtered_duration = self._read_manifest(manifest_path, config)

        if len(filtered_entries) > 0:
            print(f"Filtered {len(filtered_entries)} files which amounts to {filtered_duration} seconds of audio.")
        print(
            f"After filtering, manifest has {len(entries)} files which amounts to {total_duration} seconds of audio."
        )

        if len(entries) == 0:
            print("No tarred dataset was created as there were 0 valid samples after filtering!")
            return
        if config.shuffle:
            random.seed(config.shuffle_seed)
            print("Shuffling...")
            if config.keep_files_together:
                filename_entries = defaultdict(list)
                for ent in entries:
                    filename_entries[ent["audio_filepath"]].append(ent)
                filenames = list(filename_entries.keys())
                random.shuffle(filenames)
                shuffled_entries = []
                for filename in filenames:
                    shuffled_entries += filename_entries[filename]
                entries = shuffled_entries
            else:
                random.shuffle(entries)

        # Create shards and updated manifest entries
        print(f"Number of samples added : {len(entries)}")
        print(f"Remainder: {len(entries) % config.num_shards}")

        start_indices = []
        end_indices = []
        # Build indices
        for i in range(config.num_shards):
            start_idx = (len(entries) // config.num_shards) * i
            end_idx = start_idx + (len(entries) // config.num_shards)
            print(f"Shard {i} has entries {start_idx} ~ {end_idx}")
            files = set()
            for ent_id in range(start_idx, end_idx):
                files.add(entries[ent_id]["audio_filepath"])
            print(f"Shard {i} contains {len(files)} files")
            if i == config.num_shards - 1:
                # We discard in order to have the same number of entries per shard.
                print(f"Have {len(entries) - end_idx} entries left over that will be discarded.")

            start_indices.append(start_idx)
            end_indices.append(end_idx)

        manifest_folder, _ = os.path.split(manifest_path)

        with Parallel(n_jobs=num_workers, verbose=config.num_shards) as parallel:
            # Call parallel tarfile construction
            new_entries_list = parallel(
                delayed(self._create_shard)(entries[start_idx:end_idx], target_dir, i, manifest_folder)
                for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices))
            )

        if config.shard_manifests:
            sharded_manifests_dir = target_dir + '/sharded_manifests'
            if not os.path.exists(sharded_manifests_dir):
                os.makedirs(sharded_manifests_dir)

            for manifest in new_entries_list:
                shard_id = manifest[0]['shard_id']
                new_manifest_shard_path = os.path.join(sharded_manifests_dir, f'manifest_{shard_id}.json')
                with open(new_manifest_shard_path, 'w', encoding='utf-8') as m2:
                    for entry in manifest:
                        json.dump(entry, m2, ensure_ascii=False)
                        m2.write('\n')

        # Flatten the list of list of entries to a list of entries
        new_entries = [sample for manifest in new_entries_list for sample in manifest]
        del new_entries_list

        print("Total number of entries in manifest :", len(new_entries))

        # Write manifest
        new_manifest_path = os.path.join(target_dir, 'tarred_audio_manifest.json')
        with open(new_manifest_path, 'w', encoding='utf-8') as m2:
            for entry in new_entries:
                json.dump(entry, m2, ensure_ascii=False)
                m2.write('\n')

        # Write metadata (default metadata for new datasets)
        new_metadata_path = os.path.join(target_dir, 'metadata.yaml')
        metadata = ASRTarredDatasetMetadata()

        # Update metadata
        metadata.dataset_config = config
        metadata.num_samples_per_shard = len(new_entries) // config.num_shards

        if args.buckets_num <= 1:
            # Estimate and update dynamic bucketing args
            bucketing_kwargs = self.estimate_dynamic_bucketing_duration_bins(
                new_manifest_path, num_buckets=args.dynamic_buckets_num
            )
            for k, v in bucketing_kwargs.items():
                setattr(metadata.dataset_config, k, v)

        # Write metadata
        metadata_yaml = OmegaConf.structured(metadata)
        OmegaConf.save(metadata_yaml, new_metadata_path, resolve=True)

    def estimate_dynamic_bucketing_duration_bins(self, manifest_path: str, num_buckets: int = 30) -> dict:
        from lhotse import CutSet
        from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets
        from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

        cuts = CutSet(LazyNeMoIterator(manifest_path, metadata_only=True))
        bins = estimate_duration_buckets(cuts, num_buckets=num_buckets)
        print(
            f"Note: we estimated the optimal bucketing duration bins for {num_buckets} buckets. "
            "You can enable dynamic bucketing by setting the following options in your training script:\n"
            "  use_lhotse=true\n"
            "  use_bucketing=true\n"
            f"  num_buckets={num_buckets}\n"
            f"  bucket_duration_bins=[{','.join(map(str, bins))}]\n"
            "  batch_duration=<tune-this-value>\n"
            "If you'd like to use a different number of buckets, re-estimate this option manually using "
            "scripts/speech_recognition/estimate_duration_bins.py"
        )
        return dict(
            use_lhotse=True,
            use_bucketing=True,
            num_buckets=num_buckets,
            bucket_duration_bins=list(map(float, bins)),  # np.float -> float for YAML serialization
        )

    def create_concatenated_dataset(
        self,
        base_manifest_path: str,
        manifest_paths: List[str],
        metadata: ASRTarredDatasetMetadata,
        target_dir: str = "./tarred_concatenated/",
        num_workers: int = 1,
    ):
        """
        Creates new tarfiles in order to create a concatenated dataset, whose manifest contains the data for
        both the original dataset as well as the new data submitted in manifest paths.

        Args:
            base_manifest_path: Path to the manifest file which contains the information for the original
                tarred dataset (with flattened paths).
            manifest_paths: List of one or more paths to manifest files that will be concatenated with above
                base tarred dataset.
            metadata: ASRTarredDatasetMetadata dataclass instance with overrides from command line.
            target_dir: Output directory

        Output:
            Writes tarfiles which with indices mapping to a "concatenated" tarred dataset,
            along with the tarred dataset compatible manifest file which includes information
            about all the datasets that comprise the concatenated dataset.

            Also preserves a record of the metadata used to construct this tarred dataset.
        """
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if base_manifest_path is None:
            raise FileNotFoundError("Base manifest filepath cannot be None !")

        if manifest_paths is None or len(manifest_paths) == 0:
            raise FileNotFoundError("List of additional manifest filepaths cannot be None !")

        config = ASRTarredDatasetConfig(**(metadata.dataset_config))

        # Read the existing manifest (no filtering here)
        base_entries, _, _, _ = self._read_manifest(base_manifest_path, config)
        print(f"Read base manifest containing {len(base_entries)} samples.")

        # Precompute number of samples per shard
        if metadata.num_samples_per_shard is None:
            num_samples_per_shard = len(base_entries) // config.num_shards
        else:
            num_samples_per_shard = metadata.num_samples_per_shard

        print("Number of samples per shard :", num_samples_per_shard)

        # Compute min and max duration and update config (if no metadata passed)
        print(f"Selected max duration : {config.max_duration}")
        print(f"Selected min duration : {config.min_duration}")

        entries = []
        for new_manifest_idx in range(len(manifest_paths)):
            new_entries, total_duration, filtered_new_entries, filtered_duration = self._read_manifest(
                manifest_paths[new_manifest_idx], config
            )

            if len(filtered_new_entries) > 0:
                print(
                    f"Filtered {len(filtered_new_entries)} files which amounts to {filtered_duration:0.2f}"
                    f" seconds of audio from manifest {manifest_paths[new_manifest_idx]}."
                )
            print(
                f"After filtering, manifest has {len(entries)} files which amounts to {total_duration} seconds of audio."
            )

            entries.extend(new_entries)

        if len(entries) == 0:
            print("No tarred dataset was created as there were 0 valid samples after filtering!")
            return

        if config.shuffle:
            random.seed(config.shuffle_seed)
            print("Shuffling...")
            random.shuffle(entries)

        # Drop last section of samples that cannot be added onto a chunk
        drop_count = len(entries) % num_samples_per_shard
        total_new_entries = len(entries)
        entries = entries[:-drop_count]

        print(
            f"Dropping {drop_count} samples from total new samples {total_new_entries} since they cannot "
            f"be added into a uniformly sized chunk."
        )

        # Create shards and updated manifest entries
        num_added_shards = len(entries) // num_samples_per_shard

        print(f"Number of samples in base dataset : {len(base_entries)}")
        print(f"Number of samples in additional datasets : {len(entries)}")
        print(f"Number of added shards : {num_added_shards}")
        print(f"Remainder: {len(entries) % num_samples_per_shard}")

        start_indices = []
        end_indices = []
        shard_indices = []
        for i in range(num_added_shards):
            start_idx = (len(entries) // num_added_shards) * i
            end_idx = start_idx + (len(entries) // num_added_shards)
            shard_idx = i + config.num_shards
            print(f"Shard {shard_idx} has entries {start_idx + len(base_entries)} ~ {end_idx + len(base_entries)}")

            start_indices.append(start_idx)
            end_indices.append(end_idx)
            shard_indices.append(shard_idx)

        manifest_folder, _ = os.path.split(base_manifest_path)

        with Parallel(n_jobs=num_workers, verbose=num_added_shards) as parallel:
            # Call parallel tarfile construction
            new_entries_list = parallel(
                delayed(self._create_shard)(entries[start_idx:end_idx], target_dir, shard_idx, manifest_folder)
                for i, (start_idx, end_idx, shard_idx) in enumerate(zip(start_indices, end_indices, shard_indices))
            )

        if config.shard_manifests:
            sharded_manifests_dir = target_dir + '/sharded_manifests'
            if not os.path.exists(sharded_manifests_dir):
                os.makedirs(sharded_manifests_dir)

            for manifest in new_entries_list:
                shard_id = manifest[0]['shard_id']
                new_manifest_shard_path = os.path.join(sharded_manifests_dir, f'manifest_{shard_id}.json')
                with open(new_manifest_shard_path, 'w', encoding='utf-8') as m2:
                    for entry in manifest:
                        json.dump(entry, m2, ensure_ascii=False)
                        m2.write('\n')

        # Flatten the list of list of entries to a list of entries
        new_entries = [sample for manifest in new_entries_list for sample in manifest]
        del new_entries_list

        # Write manifest
        if metadata is None:
            new_version = 1  # start with `1`, where `0` indicates the base manifest + dataset
        else:
            new_version = metadata.version + 1

        print("Total number of entries in manifest :", len(base_entries) + len(new_entries))

        new_manifest_path = os.path.join(target_dir, f'tarred_audio_manifest_version_{new_version}.json')
        with open(new_manifest_path, 'w', encoding='utf-8') as m2:
            # First write all the entries of base manifest
            for entry in base_entries:
                json.dump(entry, m2, ensure_ascii=False)
                m2.write('\n')

            # Finally write the new entries
            for entry in new_entries:
                json.dump(entry, m2, ensure_ascii=False)
                m2.write('\n')

        # Preserve historical metadata
        base_metadata = metadata

        # Write metadata (updated metadata for concatenated datasets)
        new_metadata_path = os.path.join(target_dir, f'metadata_version_{new_version}.yaml')
        metadata = ASRTarredDatasetMetadata()

        # Update config
        config.num_shards = config.num_shards + num_added_shards

        # Update metadata
        metadata.version = new_version
        metadata.dataset_config = config
        metadata.num_samples_per_shard = num_samples_per_shard
        metadata.is_concatenated_manifest = True
        metadata.created_datetime = metadata.get_current_datetime()

        # Attach history
        current_metadata = OmegaConf.structured(base_metadata.history)
        metadata.history = current_metadata

        # Write metadata
        metadata_yaml = OmegaConf.structured(metadata)
        OmegaConf.save(metadata_yaml, new_metadata_path, resolve=True)

    def _read_manifest(self, manifest_path: str, config: ASRTarredDatasetConfig):
        """Read and filters data from the manifest"""
        # Read the existing manifest
        entries = []
        total_duration = 0.0
        filtered_entries = []
        filtered_duration = 0.0
        with open(manifest_path, 'r', encoding='utf-8') as m:
            for line in m:
                entry = json.loads(line)
                audio_key = "audio_filepath" if "audio_filepath" in entry else "audio_file"
                if audio_key not in entry:
                    raise KeyError(f"Manifest entry does not contain 'audio_filepath' or  'audio_file' key: {entry}")
                audio_filepath = entry[audio_key]
                if not os.path.isfile(audio_filepath) and not os.path.isabs(audio_filepath):
                    audio_filepath_abs = os.path.join(os.path.dirname(manifest_path), audio_filepath)
                    if not os.path.isfile(audio_filepath_abs):
                        raise FileNotFoundError(f"Could not find {audio_filepath} or {audio_filepath_abs}!")
                    entry[audio_key] = audio_filepath_abs
                if (config.max_duration is None or entry['duration'] < config.max_duration) and (
                    config.min_duration is None or entry['duration'] >= config.min_duration
                ):
                    entries.append(entry)
                    total_duration += entry["duration"]
                else:
                    filtered_entries.append(entry)
                    filtered_duration += entry['duration']

        return entries, total_duration, filtered_entries, filtered_duration

    def _write_to_tar(self, tar, audio_filepath: str, squashed_filename: str) -> None:
        if (codec := self.config.force_codec) is None or audio_filepath.endswith(f".{codec}"):
            # Add existing file without transcoding.
            tar.add(audio_filepath, arcname=squashed_filename)
        else:
            # Transcode to the desired format in-memory and add the result to the tar file.
            audio, sampling_rate = soundfile.read(audio_filepath, dtype=np.float32)
            encoded_audio = BytesIO()
            if codec == "opus":
                kwargs = {"format": "ogg", "subtype": "opus"}
            else:
                kwargs = {"format": codec}
            soundfile.write(encoded_audio, audio, sampling_rate, closefd=False, **kwargs)
            encoded_squashed_filename = f"{squashed_filename.split('.')[0]}.{codec}"
            ti = tarfile.TarInfo(encoded_squashed_filename)
            encoded_audio.seek(0)
            ti.size = len(encoded_audio.getvalue())
            tar.addfile(ti, encoded_audio)

    def _create_shard(self, entries, target_dir, shard_id, manifest_folder):
        """Creates a tarball containing the audio files from `entries`."""
        if self.config.sort_in_shards:
            entries.sort(key=lambda x: x["duration"], reverse=False)

        new_entries = []
        tar = tarfile.open(os.path.join(target_dir, f'audio_{shard_id}.tar'), mode='w', dereference=True)

        count = dict()
        for entry in entries:
            # We squash the filename since we do not preserve directory structure of audio files in the tarball.
            if os.path.exists(entry["audio_filepath"]):
                audio_filepath = entry["audio_filepath"]
            else:
                audio_filepath = os.path.join(manifest_folder, entry["audio_filepath"])
                if not os.path.exists(audio_filepath):
                    raise FileNotFoundError(f"Could not find {entry['audio_filepath']}!")

            base, ext = os.path.splitext(audio_filepath)
            base = base.replace('/', '_')
            # Need the following replacement as long as WebDataset splits on first period
            base = base.replace('.', '_')
            squashed_filename = f'{base}{ext}'
            if squashed_filename not in count:
                self._write_to_tar(tar, audio_filepath, squashed_filename)
                to_write = squashed_filename
                count[squashed_filename] = 1
            else:
                to_write = base + "-sub" + str(count[squashed_filename]) + ext
                count[squashed_filename] += 1

            # Carry over every key in the entry, override audio_filepath and shard_id
            new_entry = {
                **entry,
                'audio_filepath': to_write,
                'shard_id': shard_id,  # Keep shard ID for recordkeeping
            }
            new_entries.append(new_entry)

        tar.close()
        return new_entries

    @classmethod
    def setup_history(cls, base_metadata: ASRTarredDatasetMetadata, history: List[Any]):
        if 'history' in base_metadata.keys():
            for history_val in base_metadata.history:
                cls.setup_history(history_val, history)

        if base_metadata is not None:
            metadata_copy = copy.deepcopy(base_metadata)
            with open_dict(metadata_copy):
                metadata_copy.pop('history', None)
            history.append(metadata_copy)


def main():
    if args.buckets_num > 1:
        bucket_length = (args.max_duration - args.min_duration) / float(args.buckets_num)
        for i in range(args.buckets_num):
            min_duration = args.min_duration + i * bucket_length
            max_duration = min_duration + bucket_length
            if i == args.buckets_num - 1:
                # add a small number to cover the samples with exactly duration of max_duration in the last bucket.
                max_duration += 1e-5
            target_dir = os.path.join(args.target_dir, f"bucket{i+1}")
            print(f"Creating bucket {i+1} with min_duration={min_duration} and max_duration={max_duration} ...")
            print(f"Results are being saved at: {target_dir}.")
            create_tar_datasets(min_duration=min_duration, max_duration=max_duration, target_dir=target_dir)
            print(f"Bucket {i+1} is created.")
    else:
        create_tar_datasets(min_duration=args.min_duration, max_duration=args.max_duration, target_dir=args.target_dir)


def create_tar_datasets(min_duration: float, max_duration: float, target_dir: str):
    builder = ASRTarredDatasetBuilder()

    shard_manifests = False if args.no_shard_manifests else True

    if args.write_metadata:
        metadata = ASRTarredDatasetMetadata()
        dataset_cfg = ASRTarredDatasetConfig(
            num_shards=args.num_shards,
            shuffle=args.shuffle,
            max_duration=max_duration,
            min_duration=min_duration,
            shuffle_seed=args.shuffle_seed,
            sort_in_shards=args.sort_in_shards,
            shard_manifests=shard_manifests,
            keep_files_together=args.keep_files_together,
            force_codec=args.force_codec,
        )
        metadata.dataset_config = dataset_cfg

        output_path = os.path.join(target_dir, 'default_metadata.yaml')
        OmegaConf.save(metadata, output_path, resolve=True)
        print(f"Default metadata written to {output_path}")
        exit(0)

    if args.concat_manifest_paths is None or len(args.concat_manifest_paths) == 0:
        print("Creating new tarred dataset ...")

        # Create a tarred dataset from scratch
        config = ASRTarredDatasetConfig(
            num_shards=args.num_shards,
            shuffle=args.shuffle,
            max_duration=max_duration,
            min_duration=min_duration,
            shuffle_seed=args.shuffle_seed,
            sort_in_shards=args.sort_in_shards,
            shard_manifests=shard_manifests,
            keep_files_together=args.keep_files_together,
            force_codec=args.force_codec,
        )
        builder.configure(config)
        builder.create_new_dataset(manifest_path=args.manifest_path, target_dir=target_dir, num_workers=args.workers)

    else:
        if args.buckets_num > 1:
            raise ValueError("Concatenation feature does not support buckets_num > 1.")
        print("Concatenating multiple tarred datasets ...")

        # Implicitly update config from base details
        if args.metadata_path is not None:
            metadata = ASRTarredDatasetMetadata.from_file(args.metadata_path)
        else:
            raise ValueError("`metadata` yaml file path must be provided!")

        # Preserve history
        history = []
        builder.setup_history(OmegaConf.structured(metadata), history)
        metadata.history = history

        # Add command line overrides (everything other than num_shards)
        metadata.dataset_config.max_duration = max_duration
        metadata.dataset_config.min_duration = min_duration
        metadata.dataset_config.shuffle = args.shuffle
        metadata.dataset_config.shuffle_seed = args.shuffle_seed
        metadata.dataset_config.sort_in_shards = args.sort_in_shards
        metadata.dataset_config.shard_manifests = shard_manifests

        builder.configure(metadata.dataset_config)

        # Concatenate a tarred dataset onto a previous one
        builder.create_concatenated_dataset(
            base_manifest_path=args.manifest_path,
            manifest_paths=args.concat_manifest_paths,
            metadata=metadata,
            target_dir=target_dir,
            num_workers=args.workers,
        )

    if DALI_INDEX_SCRIPT_AVAILABLE and dali_index.INDEX_CREATOR_AVAILABLE:
        print("Constructing DALI Tarfile Index - ", target_dir)
        index_config = dali_index.DALITarredIndexConfig(tar_dir=target_dir, workers=args.workers)
        dali_index.main(index_config)


if __name__ == "__main__":
    main()
