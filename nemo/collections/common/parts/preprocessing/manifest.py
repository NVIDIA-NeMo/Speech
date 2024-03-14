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

import json
import os
import re
from collections import defaultdict
from os.path import expanduser
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from nemo.utils import logging
from nemo.utils.data_utils import DataStoreObject, datastore_path_to_local_path, is_datastore_path
from nemo.utils.nemo_logging import LogMode


class ManifestBase:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "This class is deprecated, look at https://github.com/NVIDIA/NeMo/pull/284 for correct behaviour."
        )


class ManifestEN:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "This class is deprecated, look at https://github.com/NVIDIA/NeMo/pull/284 for correct behaviour."
        )


def item_iter(
    manifests_files: Union[str, List[str]], parse_func: Callable[[str, Optional[str]], Dict[str, Any]] = None
) -> Iterator[Dict[str, Any]]:
    """Iterate through json lines of provided manifests.

    NeMo ASR pipelines often assume certain manifest files structure. In
    particular, each manifest file should consist of line-per-sample files with
    each line being correct json dict. Each such json dict should have a field
    for audio file string, a field for duration float and a field for text
    string. Offset also could be additional field and is set to None by
    default.

    Args:
        manifests_files: Either single string file or list of such -
            manifests to yield items from.

        parse_func: A callable function which accepts as input a single line
            of a manifest and optionally the manifest file itself,
            and parses it, returning a dictionary mapping from str -> Any.

    Yields:
        Parsed key to value item dicts.

    Raises:
        ValueError: If met invalid json line structure.
    """

    if isinstance(manifests_files, str):
        manifests_files = [manifests_files]

    if parse_func is None:
        parse_func = __parse_item

    errors = defaultdict(list)
    k = -1
    logging.debug('Manifest files: %s', str(manifests_files))
    for manifest_file in manifests_files:
        logging.debug('Using manifest file: %s', str(manifest_file))
        cached_manifest_file = DataStoreObject(manifest_file).get()
        logging.debug('Cached at: %s', str(cached_manifest_file))
        with open(expanduser(cached_manifest_file), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                k += 1
                try:
                    item = parse_func(line, manifest_file)
                except json.JSONDecodeError:
                    errors[str(manifest_file)].append(line)
                    continue
                item['id'] = k

                yield item

    if len(errors) > 0:
        for filename, lines in errors.items():
            logging.error("=============================================")
            logging.error(f"Failed to parse {len(lines)} lines from manifest file: {filename}")
            for line in lines:
                logging.error(f"-- Failed to parse line: `{line}`")
        raise RuntimeError("Failed to parse some lines from manifest files. See logs for more details.")


def __parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
    item = json.loads(line)

    # Audio file
    if 'audio_filename' in item:
        item['audio_file'] = item.pop('audio_filename')
    elif 'audio_filepath' in item:
        item['audio_file'] = item.pop('audio_filepath')

    # Video File
    if 'video_filename' in item:
        item['video_file'] = item.pop('video_filename')
    elif 'video_filepath' in item:
        item['video_file'] = item.pop('video_filepath')

    if 'video_file' not in item and 'audio_file' not in item:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper audio/video file key."
        )

    # If the audio/video path is a relative path and does not exist,
    # try to attach the parent directory of manifest to the audio path.
    # Revert to the original path if the new path still doesn't exist.
    # Assume that the audio path is like "wavs/xxxxxx.wav".
    if 'audio_file' in item:
        item['audio_file'] = get_full_path(audio_file=item['audio_file'], manifest_file=manifest_file)
    if 'video_file' in item:
        item['video_file'] = get_full_path(audio_file=item['video_file'], manifest_file=manifest_file)

    # Duration.
    if 'duration' not in item:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper duration key."
        )

    # Text.
    if 'text' in item:
        pass
    elif 'text_filepath' in item:
        with open(item.pop('text_filepath'), 'r') as f:
            item['text'] = f.read().replace('\n', '')
    elif 'normalized_text' in item:
        item['text'] = item['normalized_text']
    else:
        item['text'] = ""

    # Optional RTTM file
    if 'rttm_file' in item:
        pass
    elif 'rttm_filename' in item:
        item['rttm_file'] = item.pop('rttm_filename')
    elif 'rttm_filepath' in item:
        item['rttm_file'] = item.pop('rttm_filepath')
    else:
        item['rttm_file'] = None
    if item['rttm_file'] is not None:
        item['rttm_file'] = get_full_path(audio_file=item['rttm_file'], manifest_file=manifest_file)

    # Optional audio feature file
    if 'feature_file' in item:
        pass
    elif 'feature_filename' in item:
        item['feature_file'] = item.pop('feature_filename')
    elif 'feature_filepath' in item:
        item['feature_file'] = item.pop('feature_filepath')
    else:
        item['feature_file'] = None
    if item['feature_file'] is not None:
        item['feature_file'] = get_full_path(audio_file=item['feature_file'], manifest_file=manifest_file)

    item = dict(
        audio_file=item.get('audio_file', None),
        video_file=item.get('video_file', None),
        duration=item['duration'],
        text=item['text'],
        rttm_file=item['rttm_file'],
        feature_file=item['feature_file'],
        offset=item.get('offset', None),
        speaker=item.get('speaker', None),
        orig_sr=item.get('orig_sample_rate', None),
        token_labels=item.get('token_labels', None),
        lang=item.get('lang', None),
    )
    return item


def is_tarred_dataset(audio_file: str, manifest_file: Optional[str] = None) -> bool:
    if "/" in audio_file or manifest_file is None:
        # audio files in a tarred dataset don't have `/` in their paths
        return False
    if os.path.basename(manifest_file) == "tarred_audio_manifest.json":
        # the manifest file is a tarred manifest
        return True
    if "/sharded_manifests/" in manifest_file and re.match(r'^manifest_(\d+)\.json$', os.path.basename(manifest_file)):
        # the manifest file is a sharded manifest
        return True
    return False


def get_full_path(
    audio_file: Union[str, List[str]],
    manifest_file: Optional[str] = None,
    data_dir: Optional[str] = None,
    audio_file_len_limit: int = 255,
) -> Union[str, List[str]]:
    """Get full path to audio_file.

    If the audio_file is a relative path and does not exist,
    try to attach the parent directory of manifest to the audio path.
    Revert to the original path if the new path still doesn't exist.
    Assume that the audio path is like "wavs/xxxxxx.wav".

    Args:
        audio_file: path to an audio file, either absolute or assumed relative
                    to the manifest directory or data directory.
                    Alternatively, a list of paths may be provided.
        manifest_file: path to a manifest file
        data_dir: path to a directory containing data, use only if a manifest file is not provided
        audio_file_len_limit: limit for length of audio_file when using relative paths

    Returns:
        Full path to audio_file or a list of paths.
    """
    if isinstance(audio_file, list):
        # If input is a list, return a list of full paths
        return [
            get_full_path(
                audio_file=a_file,
                manifest_file=manifest_file,
                data_dir=data_dir,
                audio_file_len_limit=audio_file_len_limit,
            )
            for a_file in audio_file
        ]
    elif isinstance(audio_file, str):
        # If input is a string, get the corresponding full path
        if is_tarred_dataset(audio_file=audio_file, manifest_file=manifest_file):
            logging.warning(
                f"Manifest file `{manifest_file}` seems to be part of a tarred dataset, skip checking for relative paths. If this is not intended, please avoid having `/sharded_manifests/` and `tarred_audio_manifest.json` in manifest_filepath.",
                mode=LogMode.ONCE,
            )
            return audio_file
        if (
            (len(audio_file) < audio_file_len_limit)
            and not os.path.isabs(audio_file)
            and not os.path.isfile(audio_file)
        ):
            # If audio_file is not available and the path is not absolute, the full path is assumed
            # to be relative to the manifest file parent directory or data directory.
            if manifest_file is None and data_dir is None:
                raise ValueError(f'Use either manifest_file or data_dir to specify the data directory.')
            elif manifest_file is not None and data_dir is not None:
                raise ValueError(
                    f'Parameters manifest_file and data_dir cannot be used simultaneously. Currently manifest_file is {manifest_file} and data_dir is {data_dir}.'
                )

            # resolve the data directory
            if data_dir is None:
                data_dir = os.path.dirname(manifest_file)

            # assume audio_file path is relative to data_dir
            audio_file_path = os.path.join(data_dir, audio_file)

            if is_datastore_path(audio_file_path):
                # If audio was originally on an object store, use locally-cached path
                audio_file_path = datastore_path_to_local_path(audio_file_path)

            if os.path.isfile(audio_file_path):
                audio_file = os.path.abspath(audio_file_path)
            else:
                audio_file = expanduser(audio_file)
        else:
            audio_file = expanduser(audio_file)
        return audio_file
    else:
        raise ValueError(f'Unexpected audio_file type {type(audio_file)}, audio_file {audio_file}.')
