# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script is used to preprocess audio before TTS model training.

It can be configured to do several processing steps such as silence trimming, volume normalization,
and duration filtering.

These can be done separately through multiple executions of the script, or all at once to avoid saving
too many copies of the same audio.

Most of these can also be done by the TTS data loader at training time, but doing them ahead of time
lets us implement more complex processing, validate the corectness of the output, and save on compute time.

$ HYDRA_FULL_ERROR=1 python <nemo_root_path>/scripts/dataset_processing/tts/audio_processing/preprocess_audio.py \
    --config-path=<nemo_root_path>/scripts/dataset_processing/tts/audio_processing/config \
    --config-name=preprocessing.yaml \
    data_base_dir="/home/data" \
    config.num_workers=1
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import soundfile as sf
from hydra.utils import instantiate
from joblib import Parallel, delayed
from tqdm import tqdm

from nemo.collections.tts.parts.preprocessing.audio_trimming import AudioTrimmer
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    get_base_dir,
    normalize_volume,
    read_manifest,
    write_manifest,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class AudioPreprocessingConfig:
    # Input training manifest.
    input_manifest: Path
    # New training manifest after processing audio.
    output_manifest: Path
    # Directory to save processed audio to.
    output_dir: Path
    # Number of threads to use. -1 will use all available CPUs.
    num_workers: int = -1
    # If provided, maximum number of entries in the manifest to process.
    max_entries: int = 0
    # If provided, rate to resample the audio to.
    output_sample_rate: int = 0
    # If provided, peak volume to normalize audio to.
    volume_level: float = 0.0
    # If provided, filter out utterances shorter than min_duration.
    min_duration: float = 0.0
    # If provided, filter out utterances longer than min_duration.
    max_duration: float = float("inf")
    # If provided, output filter_file will contain list of utterances filtered out.
    filter_file: Path = None


def _process_entry(
    entry: dict,
    base_dir: Path,
    output_dir: Path,
    audio_trimmer: AudioTrimmer,
    output_sample_rate: int,
    volume_level: float,
) -> Tuple[dict, float, float]:
    audio_filepath = Path(entry["audio_filepath"])
    rel_audio_path = audio_filepath.relative_to(base_dir)
    input_path = os.path.join(base_dir, rel_audio_path)
    output_path = os.path.join(output_dir, rel_audio_path)

    audio, sample_rate = librosa.load(input_path, sr=None)

    if audio_trimmer is not None:
        audio_id = str(audio_filepath)
        audio, start_i, end_i = audio_trimmer.trim_audio(audio=audio, sample_rate=sample_rate, audio_id=audio_id)

    if output_sample_rate is not None:
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=output_sample_rate)
        sample_rate = output_sample_rate

    if volume_level:
        audio = normalize_volume(audio, volume_level=volume_level)

    sf.write(file=output_path, data=audio, samplerate=sample_rate)

    original_duration = librosa.get_duration(filename=str(audio_filepath))
    output_duration = librosa.get_duration(filename=str(output_path))

    entry["audio_filepath"] = output_path
    entry["duration"] = output_duration

    return entry, original_duration, output_duration


@hydra_runner(config_path='config', config_name='preprocessing')
def main(cfg):
    config = instantiate(cfg.config)
    logging.info(f"Running audio preprocessing with config: {config}")

    input_manifest_path = Path(config.input_manifest)
    output_manifest_path = Path(config.output_manifest)
    output_dir = Path(config.output_dir)
    num_workers = config.num_workers
    max_entries = config.max_entries
    output_sample_rate = config.output_sample_rate
    volume_level = config.volume_level
    min_duration = config.min_duration
    max_duration = config.max_duration
    filter_file = Path(config.filter_file)

    if cfg.trim:
        audio_trimmer = instantiate(cfg.trim)
    else:
        audio_trimmer = None

    output_dir.mkdir(exist_ok=True, parents=True)

    entries = read_manifest(input_manifest_path)
    if max_entries:
        entries = entries[:max_entries]

    audio_paths = [entry["audio_filepath"] for entry in entries]
    base_dir = get_base_dir(audio_paths)

    # 'threading' backend is required when parallelizing torch models.
    job_outputs = Parallel(n_jobs=num_workers, backend='threading')(
        delayed(_process_entry)(
            entry=entry,
            base_dir=base_dir,
            output_dir=output_dir,
            audio_trimmer=audio_trimmer,
            output_sample_rate=output_sample_rate,
            volume_level=volume_level,
        )
        for entry in tqdm(entries)
    )

    output_entries = []
    filtered_entries = []
    original_durations = 0.0
    output_durations = 0.0
    for output_entry, original_duration, output_duration in job_outputs:

        if not min_duration <= output_duration <= max_duration:
            if output_duration != original_duration:
                output_entry["original_duration"] = original_duration
            filtered_entries.append(output_entry)
            continue

        original_durations += original_duration
        output_durations += output_duration
        output_entries.append(output_entry)

    write_manifest(manifest_path=output_manifest_path, entries=output_entries)
    if filter_file:
        write_manifest(manifest_path=filter_file, entries=filtered_entries)

    logging.info(f"Duration of original audio: {original_durations / 3600} hours")
    logging.info(f"Duration of processed audio: {output_durations / 3600} hours")


if __name__ == "__main__":
    main()
