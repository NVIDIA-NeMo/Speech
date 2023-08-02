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
lets us implement more complex processing, validate the correctness of the output, and save on compute time.

$ python <nemo_root_path>/scripts/dataset_processing/tts/preprocess_audio.py \
    --input_manifest="<data_root_path>/manifest.json" \
    --output_manifest="<data_root_path>/manifest_processed.json" \
    --input_audio_dir="<data_root_path>/audio" \
    --output_audio_dir="<data_root_path>/audio_processed" \
    --num_workers=1 \
    --trim_config_path="<nemo_root_path>/examples/tts/conf/trim/energy.yaml" \
    --output_sample_rate=22050 \
    --output_format=flac \
    --volume_level=0.95 \
    --min_duration=0.5 \
    --max_duration=20.0 \
    --filter_file="filtered.txt"
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import librosa
import soundfile as sf
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.tts.parts.preprocessing.audio_trimming import AudioTrimmer
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_abs_rel_paths, normalize_volume
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute speaker level pitch statistics.",
    )
    parser.add_argument(
        "--input_manifest", required=True, type=Path, help="Path to input training manifest.",
    )
    parser.add_argument(
        "--input_audio_dir", required=True, type=Path, help="Path to base directory with audio files.",
    )
    parser.add_argument(
        "--output_manifest", required=True, type=Path, help="Path to output training manifest with processed audio.",
    )
    parser.add_argument(
        "--output_audio_dir", required=True, type=Path, help="Path to output directory for audio files.",
    )
    parser.add_argument(
        "--overwrite_audio",
        action=argparse.BooleanOptionalAction,
        help="Whether to reprocess and overwrite existing audio files in output_audio_dir.",
    )
    parser.add_argument(
        "--overwrite_manifest",
        action=argparse.BooleanOptionalAction,
        help="Whether to overwrite the output manifest file if it exists.",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of parallel threads to use. If -1 all CPUs are used."
    )
    parser.add_argument(
        "--trim_config_path",
        required=False,
        type=Path,
        help="Path to config file for nemo.collections.tts.data.audio_trimming.AudioTrimmer",
    )
    parser.add_argument(
        "--max_entries", default=0, type=int, help="If provided, maximum number of entries in the manifest to process."
    )
    parser.add_argument(
        "--output_sample_rate", default=0, type=int, help="If provided, rate to resample the audio to."
    )
    parser.add_argument(
        "--output_format",
        default="wav",
        type=str,
        help="If provided, format output audio will be saved as. If not provided, will keep original format.",
    )
    parser.add_argument(
        "--volume_level", default=0.0, type=float, help="If provided, peak volume to normalize audio to."
    )
    parser.add_argument(
        "--min_duration", default=0.0, type=float, help="If provided, filter out utterances shorter than min_duration."
    )
    parser.add_argument(
        "--max_duration", default=0.0, type=float, help="If provided, filter out utterances longer than max_duration."
    )
    parser.add_argument(
        "--filter_file",
        required=False,
        type=Path,
        help="If provided, output filter_file will contain list of " "utterances filtered out.",
    )
    args = parser.parse_args()
    return args


def _process_entry(
    entry: dict,
    input_audio_dir: Path,
    output_audio_dir: Path,
    overwrite_audio: bool,
    audio_trimmer: AudioTrimmer,
    output_sample_rate: int,
    output_format: str,
    volume_level: float,
) -> Tuple[dict, float, float]:
    audio_filepath = Path(entry["audio_filepath"])

    audio_path, audio_path_rel = get_abs_rel_paths(input_path=audio_filepath, base_path=input_audio_dir)

    if not output_format:
        output_format = audio_path.suffix

    output_path = output_audio_dir / audio_path_rel
    output_path = output_path.with_suffix(output_format)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if output_path.exists() and not overwrite_audio:
        original_duration = librosa.get_duration(path=audio_path)
        output_duration = librosa.get_duration(path=output_path)
    else:
        audio, sample_rate = librosa.load(audio_path, sr=None)
        original_duration = librosa.get_duration(y=audio, sr=sample_rate)
        if audio_trimmer is not None:
            audio, start_i, end_i = audio_trimmer.trim_audio(
                audio=audio, sample_rate=int(sample_rate), audio_id=str(audio_path)
            )

        if output_sample_rate:
            audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=output_sample_rate)
            sample_rate = output_sample_rate

        if volume_level:
            audio = normalize_volume(audio, volume_level=volume_level)

        if audio.size > 0:
            sf.write(file=output_path, data=audio, samplerate=sample_rate)
            output_duration = librosa.get_duration(y=audio, sr=sample_rate)
        else:
            output_duration = 0.0

    entry["duration"] = round(output_duration, 2)

    if os.path.isabs(audio_filepath):
        entry["audio_filepath"] = str(output_path)
    else:
        output_filepath = audio_path_rel.with_suffix(output_format)
        entry["audio_filepath"] = str(output_filepath)

    return entry, original_duration, output_duration


def main():
    args = get_args()

    input_manifest_path = args.input_manifest
    output_manifest_path = args.output_manifest
    input_audio_dir = args.input_audio_dir
    output_audio_dir = args.output_audio_dir
    overwrite_audio = args.overwrite_audio
    overwrite_manifest = args.overwrite_manifest
    num_workers = args.num_workers
    max_entries = args.max_entries
    output_sample_rate = args.output_sample_rate
    output_format = args.output_format
    volume_level = args.volume_level
    min_duration = args.min_duration
    max_duration = args.max_duration
    filter_file = args.filter_file

    if output_manifest_path.exists():
        if overwrite_manifest:
            print(f"Will overwrite existing manifest path: {output_manifest_path}")
        else:
            raise ValueError(f"Manifest path already exists: {output_manifest_path}")

    if args.trim_config_path:
        audio_trimmer_config = OmegaConf.load(args.trim_config_path)
        audio_trimmer = instantiate(audio_trimmer_config)
    else:
        audio_trimmer = None

    if output_format:
        if output_format.upper() not in sf.available_formats():
            raise ValueError(f"Unsupported output audio format: {output_format}")
        output_format = f".{output_format}"

    output_audio_dir.mkdir(exist_ok=True, parents=True)

    entries = read_manifest(input_manifest_path)
    if max_entries:
        entries = entries[:max_entries]

    # 'threading' backend is required when parallelizing torch models.
    job_outputs = Parallel(n_jobs=num_workers, backend='threading')(
        delayed(_process_entry)(
            entry=entry,
            input_audio_dir=input_audio_dir,
            output_audio_dir=output_audio_dir,
            overwrite_audio=overwrite_audio,
            audio_trimmer=audio_trimmer,
            output_sample_rate=output_sample_rate,
            output_format=output_format,
            volume_level=volume_level,
        )
        for entry in tqdm(entries)
    )

    output_entries = []
    filtered_entries = []
    original_durations = 0.0
    output_durations = 0.0
    for output_entry, original_duration, output_duration in job_outputs:
        original_durations += original_duration

        if (
            output_duration == 0.0
            or (min_duration and output_duration < min_duration)
            or (max_duration and output_duration > max_duration)
        ):
            if output_duration != original_duration:
                output_entry["original_duration"] = original_duration
            filtered_entries.append(output_entry)
            continue

        output_durations += output_duration
        output_entries.append(output_entry)

    write_manifest(output_path=output_manifest_path, target_manifest=output_entries, ensure_ascii=False)
    if filter_file:
        write_manifest(output_path=str(filter_file), target_manifest=filtered_entries, ensure_ascii=False)

    logging.info(f"Duration of original audio: {original_durations / 3600:.2f} hours")
    logging.info(f"Duration of processed audio: {output_durations / 3600:.2f} hours")


if __name__ == "__main__":
    main()
