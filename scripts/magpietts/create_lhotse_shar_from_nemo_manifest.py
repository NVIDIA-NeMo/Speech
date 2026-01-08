# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script requires the following updates to lhotse: add `shard_offset` in lhotse's writers.
$ pip install git+https://github.com/lhotse-speech/lhotse.git@883c24b5f6cdc4bbc73e89186e99f7907262b59c

Example of manifest:
    {
        "audio_filepath": "train-clean-360/4098/11547/4098_11547_000032_000000.wav",
        "text": "\"Isn't it?\" queried Theo.",
        "speaker": "| Language:en Dataset:LibriTTS Speaker:4098 |",
        "chapter_id": "11547",
        "utter_id": "000032_000000",
        "duration": 1.9700416666666667,
        "normalized_text": "\"Isn't it?\" queried Theo.",
        "context_speaker_similarity": 0.7800518870353699,
        "context_audio_filepath": "train-clean-360/4098/11547/4098_11547_000031_000001.wav",
        "context_audio_duration": 9.45
    }

Example usage:
    python scripts/magpietts/create_lhotse_shar_from_nemo_manifest.py \
        --manifest-path ${MANIFEST} \
        --audio-base-dir ${AUDIO_BASE_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --num-jobs ${NUM_JOBS} \
        --processing-chunk-size ${CHUNK_SIZE} \
        --audio-format ${AUDIO_FORMAT} \
        --log-level ${LOG_LEVEL} \
        --shuffle \
        --shuffle-seed 42 \
    2>&1 | tee ./log/create_lhotse_shar_from_nemo_manifest.stdout

Expected output:
    $ tree ${OUTPUT_DIR}
    ${OUTPUT_DIR}/
        cuts/
            cuts.000000.jsonl.gz
            cuts.000001.jsonl.gz
            ...
        target_audio/
            recording.000000.tar
            recording.000001.tar
            ...
        context_audio/
            recording.000000.tar
            recording.000001.tar
            ...
"""

import argparse
import itertools
import logging
import math
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

from lhotse import AudioSource, MonoCut, Recording, SupervisionSegment, compute_num_samples, fastcopy
from lhotse.serialization import load_jsonl
from lhotse.shar.writers import AudioTarWriter, JsonlShardWriter
from tqdm import tqdm

NEMO_KEYS_NO_NEED_TO_LOG_IN_CUSTOM_FIELDS_FOR_SUPERVISION = [
    "audio_filepath",
    "context_audio_filepath",
    "text",
    "offset",
    "duration",
    "speaker",
]


def to_shar_placeholder(recording: Recording, cut: MonoCut) -> Recording:
    """this function is borrowed from lhotse.shar.writers.to_shar_placeholder. The only change for Recording instance is to update the id with cut.id."""
    return fastcopy(
        recording,
        id=cut.id,
        # Creates a single AudioSource out of multiple ones.
        sources=[AudioSource(type="shar", channels=recording.channel_ids, source="")],
        # Removes the transform metadata because they were already executed.
        transforms=None,
        duration=cut.duration,
        num_samples=compute_num_samples(cut.duration, recording.sampling_rate),
    )


def check_speaker_format(item: str):
    """Enforce speaker format like '| Language:en Dataset:HiFiTTS Speaker:9136_other |'"""
    pattern = r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|"
    if not isinstance(item, str):
        return False
    return bool(re.match(pattern, item))


def get_recording_id(relative_path: str) -> str:
    """Generate a recording ID from the relative audio path."""
    return "rec-" + relative_path.rsplit(".", 1)[0].replace("/", "-")


def process_manifest_entry(entry: Dict[str, Any], audio_base_dir: Path) -> Tuple[MonoCut, MonoCut] | None:
    """
    Processes a single entry from the NeMo manifest to create Lhotse objects.

    Returns:
        tuple: (target_cut, context_cut) or None if an error occurs.
    """
    try:
        # Required fields
        target_audio_path_relative = entry.get("audio_filepath")
        context_audio_path_relative = entry.get("context_audio_filepath")
        target_audio_duration = entry.get("duration")
        context_audio_duration = entry.get("context_audio_duration")
        text = entry.get("text")
        # observed cases when text is empty while normalized_text is not.
        if not text or not text.strip():
            text = entry.get("normalized_text")
        speaker = entry.get("speaker")

        # Check required fields
        if not all(
            [
                target_audio_path_relative,
                context_audio_path_relative,
                target_audio_duration,
                context_audio_duration,
                text,
                speaker,
            ]
        ):
            logging.warning(f"Skipping entry due to missing fields: {entry}")
            return None

        # Check speaker format
        if not check_speaker_format(speaker):
            logging.warning(f"Skipping entry due to incorrect speaker format: {entry}")
            return None

        target_audio_filepath = audio_base_dir / target_audio_path_relative
        context_audio_filepath = audio_base_dir / context_audio_path_relative

        if not target_audio_filepath.is_file():
            logging.warning(
                f"Skipping entry due to missing target audio file: {target_audio_filepath} from entry: {entry}"
            )
            return None
        if not context_audio_filepath.is_file():
            logging.warning(
                f"Skipping entry due to missing context audio file: {context_audio_filepath} from entry: {entry}"
            )
            return None

        # Create IDs
        target_recording_id = get_recording_id(target_audio_path_relative)
        context_recording_id = get_recording_id(context_audio_path_relative)

        # Create Recordings
        # TODO: if input is FLAC, then we should set AudioSegment.from_file(int_values=True). Does this applies to lhotse?
        target_recording = Recording.from_file(target_audio_filepath, recording_id=target_recording_id)
        context_recording = Recording.from_file(context_audio_filepath, recording_id=context_recording_id)

        # Custom fields exist in manifests, so better to keep them for future usage.
        custom_fields = {
            key: val
            for key, val in entry.items()
            if key not in NEMO_KEYS_NO_NEED_TO_LOG_IN_CUSTOM_FIELDS_FOR_SUPERVISION
        }
        custom_fields["context_recording_id"] = context_recording_id

        # Extract language from speaker string
        lang_match = re.search(r"Language:(\w+)", speaker)
        language = lang_match.group(1) if lang_match else None

        # offset in seconds
        target_offset_in_seconds = entry.get("offset", 0.0)
        context_offset_in_seconds = entry.get("context_audio_offset", 0.0)

        # Create Supervision for target cut. We constrain one supervision per cut for now.
        supervision = SupervisionSegment(
            id=f"sup-{target_recording_id}",
            recording_id=target_recording_id,
            start=target_offset_in_seconds,
            duration=target_audio_duration,  # duration from manifest
            channel=0,  # only support mono audio for now
            text=text,
            language=language,
            speaker=speaker,
            custom=custom_fields,
        )

        # Create target cut
        target_cut_id = f"cut-{target_recording_id}-{target_offset_in_seconds:.2f}-{target_audio_duration:.2f}"
        target_cut = MonoCut(
            id=target_cut_id,
            start=target_offset_in_seconds,
            duration=target_audio_duration,
            channel=0,  # only support mono audio for now
            recording=target_recording,
            supervisions=[supervision],
        )
        if not math.isclose(target_cut.duration, target_audio_duration, abs_tol=0.1):
            logging.warning(
                f"Manifest duration ({target_audio_duration}) differs significantly from cut duration ({target_cut.duration}) for {target_recording_id}. Using cut duration."
            )
            target_cut.supervisions[0].duration = target_cut.duration

        # Create context cut. This cut is only used to load segmented audio and would not be stored in the final manifest.
        context_cut_id = (
            f"context_cut-{context_recording_id}-{context_offset_in_seconds:.2f}-{context_audio_duration:.2f}"
        )
        if context_cut_id.split("-", 1)[1] == target_cut_id.split("-", 1)[1]:
            logging.warning(f"Context cut has the same recording segment as target cut. Skipping entry: {entry}")
            return None

        context_cut = MonoCut(
            id=context_cut_id,
            start=context_offset_in_seconds,
            duration=context_audio_duration,
            channel=0,  # only support mono audio for now
            recording=context_recording,
        )
        return target_cut, context_cut

    except Exception as e:
        logging.error(f"Skipping entry due to error during metadata processing: {entry}: {e}", exc_info=True)
        return None


def shuffle_jsonl_file(input_path: Path, seed: int = None) -> Path:
    """
    Shuffle lines in a JSONL file and write to a shuffled copy.

    Args:
        input_path: Path to the original JSONL file
        seed: Random seed for reproducible shuffling

    Returns:
        Path to the shuffled file
    """
    if seed is not None:
        random.seed(seed)

    logging.info(f"Reading and shuffling manifest entries from {input_path}")

    # Read all lines into memory
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    logging.info(f"Loaded {len(lines)} entries, now shuffling...")

    # Shuffle the lines
    random.shuffle(lines)

    # Create output path with "_shuffled" suffix
    shuffled_path = input_path.parent / f"{input_path.stem}_shuffled{input_path.suffix}"

    # Write shuffled content
    with open(shuffled_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    logging.info(f"Shuffled manifest written to: {shuffled_path}")
    return shuffled_path


def chunked_iterator(iterable, chunk_size):
    """Yield successive chunks from iterable."""
    _it = iter(iterable)
    while _chunk := tuple(itertools.islice(_it, chunk_size)):
        yield _chunk


def process_and_write_chunk(
    manifest_chunk_with_idx: Tuple[int, Tuple[Dict[str, Any], ...]],
    audio_base_dir: Path,
    output_dir: Path,
    audio_format: str,
) -> Dict[str, int]:
    """
    Processes a chunk of manifest entries, loads audio, and writes corresponding
    single shard files for cuts, target audio, and context audio.
    Designed to be run in a parallel worker process.
    Loads and writes audio iteratively to save memory.

    Returns a dict containing processing stats like 'processed', 'initial_errors', 'audio_load_errors'.
    """
    chunk_idx, manifest_chunk = manifest_chunk_with_idx
    worker_pid = os.getpid()
    logging.debug(f"[Worker {worker_pid}, Chunk {chunk_idx}] Starting processing {len(manifest_chunk)} entries.")

    # --- 1. Process manifest entries to get Cut objects ---
    chunk_metadata = []
    initial_errors = 0
    for entry in manifest_chunk:
        result = process_manifest_entry(entry, audio_base_dir=audio_base_dir)
        if result is not None:
            chunk_metadata.append(result)
        else:
            initial_errors += 1

    if not chunk_metadata:
        logging.warning(
            f"[Worker {worker_pid}, Chunk {chunk_idx}] No valid entries after initial processing. Skipping."
        )
        return {"processed": 0, "initial_errors": initial_errors, "audio_load_errors": 0, "write_errors": 0}

    logging.debug(
        f"[Worker {worker_pid}, Chunk {chunk_idx}] Collected {len(chunk_metadata)} cut pairs after initial processing."
    )

    # --- 2. Initialize writers and perform iterative load-and-write ---
    cuts_dir = output_dir / "cuts"
    target_recordings_dir = output_dir / "target_audio"
    context_recordings_dir = output_dir / "context_audio"

    cuts_pattern = str(cuts_dir / "cuts.%06d.jsonl.gz")
    target_rec_pattern = str(target_recordings_dir / "recording.%06d.tar")
    context_rec_pattern = str(context_recordings_dir / "recording.%06d.tar")

    chunk_processed_count = 0
    chunk_audio_load_errors = 0  # Errors during audio loading phase for this chunk
    chunk_write_errors = 0  # Errors during write phase for this chunk

    logging.debug(
        f"[Worker {worker_pid}, Chunk {chunk_idx}] Initializing writers with offset {chunk_idx} and processing {len(chunk_metadata)} pairs iteratively..."
    )
    try:
        # Specify shard_size with len(chunk_metadata) and shard_offset with chunk_idx, ensuring each chunk is written to a separate shard file.
        shard_size_for_worker = len(chunk_metadata)
        with (
            JsonlShardWriter(
                pattern=cuts_pattern, shard_size=shard_size_for_worker, shard_offset=chunk_idx
            ) as cut_writer,
            AudioTarWriter(
                pattern=target_rec_pattern,
                shard_size=shard_size_for_worker,
                format=audio_format,
                shard_offset=chunk_idx,
            ) as target_rec_writer,
            AudioTarWriter(
                pattern=context_rec_pattern,
                shard_size=shard_size_for_worker,
                format=audio_format,
                shard_offset=chunk_idx,
            ) as context_rec_writer,
        ):
            # Iterate directly over chunk_metadata
            for target_cut, context_cut in chunk_metadata:
                # 1. load target/context audio given the audio offset
                try:
                    target_audio = target_cut.load_audio()
                    context_audio = context_cut.load_audio()
                except Exception as e:
                    logging.error(
                        f"[Worker {worker_pid}, Chunk {chunk_idx}] Error loading target/context audio for cut {target_cut}: {e}",
                        exc_info=True,
                    )
                    chunk_audio_load_errors += 1
                    continue

                # 2. Write target audio and context audio
                try:
                    target_rec_writer.write(
                        key=target_cut.id,
                        value=target_audio,
                        sampling_rate=target_cut.sampling_rate,
                        manifest=to_shar_placeholder(
                            target_cut.recording, target_cut
                        ),  # update manifest.id with target_cut.id that has the audio offset and duration
                    )
                    context_rec_writer.write(
                        key=target_cut.id,  # use target cut id as key for context audio to ensure reference
                        value=context_audio,
                        sampling_rate=context_cut.sampling_rate,
                        manifest=to_shar_placeholder(
                            context_cut.recording, context_cut
                        ),  # update manifest.id with context_cut.id that has the audio offset and duration
                    )
                except Exception as e:
                    logging.error(
                        f"[Worker {worker_pid}, Chunk {chunk_idx}] Error writing target/context audio for target cut {target_cut}: {e}",
                        exc_info=True,
                    )
                    chunk_write_errors += 1
                    continue

                # 3. write cut metadata
                try:
                    cut_writer.write(target_cut)
                except Exception as e:
                    logging.error(
                        f"[Worker {worker_pid}, Chunk {chunk_idx}] Error writing cut metadata for cut {target_cut}: {e}",
                        exc_info=True,
                    )
                    chunk_write_errors += 1
                    continue

                chunk_processed_count += 1

    except Exception as e:
        logging.error(
            f"[Worker {worker_pid}, Chunk {chunk_idx}] CRITICAL error during writer initialization: {e}", exc_info=True
        )
        chunk_write_errors = len(chunk_metadata)
        chunk_processed_count = 0

    # This part is only reached if the main try block completes without critical errors
    logging.debug(
        f"[Worker {worker_pid}, Chunk {chunk_idx}] Finished chunk. Processed: {chunk_processed_count}, Audio Load Errors: {chunk_audio_load_errors}, Write Errors: {chunk_write_errors}"
    )

    return {
        "processed": chunk_processed_count,
        "initial_errors": initial_errors,  # Errors from initial metadata processing
        "audio_load_errors": chunk_audio_load_errors,
        "write_errors": chunk_write_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert NeMo manifest to sharded Lhotse JSONL/TARs using parallel workers per chunk.",
    )
    parser.add_argument("--manifest-path", required=True, type=Path, help="Path to the input NeMo JSON manifest file.")
    parser.add_argument(
        "--audio-base-dir", required=True, type=Path, help="Base directory where audio files are located."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Base directory to save the sharded outputs.")
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=max(1, os.cpu_count() // 2),
        help="Number of parallel worker processes (each processing a whole chunk/shard).",
    )
    parser.add_argument(
        "--processing-chunk-size",
        type=int,
        default=4000,
        help="Number of manifest entries per chunk (effectively the items per output shard file).",
    )
    parser.add_argument(
        "--audio-format", type=str, default="flac", help="Audio format for TAR writers (e.g., flac, wav, opus)."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the main process and workers.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the manifest entries before processing.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling (only used if --shuffle is enabled).",
    )

    args = parser.parse_args()

    # Configure logging based on argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)

    # Ensure output directories exist
    cuts_dir = args.output_dir / "cuts"
    target_recordings_dir = args.output_dir / "target_audio"
    context_recordings_dir = args.output_dir / "context_audio"
    cuts_dir.mkdir(parents=True, exist_ok=True)
    target_recordings_dir.mkdir(parents=True, exist_ok=True)
    context_recordings_dir.mkdir(parents=True, exist_ok=True)

    # Handle shuffling if requested
    if args.shuffle:
        logging.info(f"Shuffling manifest entries from: {args.manifest_path}")
        shuffled_manifest_path = shuffle_jsonl_file(args.manifest_path, seed=args.shuffle_seed)
        manifest_iterable = load_jsonl(shuffled_manifest_path)
        logging.info(f"Using shuffled manifest for processing: {shuffled_manifest_path}")
    else:
        logging.info(f"Reading NeMo manifest lazily from: {args.manifest_path}")
        manifest_iterable = load_jsonl(args.manifest_path)

    logging.info(
        f"Processing manifest in chunks of {args.processing_chunk_size} using {args.num_jobs} parallel workers..."
    )

    total_processed_count = 0
    total_initial_errors = 0
    total_audio_load_errors = 0
    total_write_errors = 0
    num_chunks = 0

    worker_func = partial(
        process_and_write_chunk,
        audio_base_dir=args.audio_base_dir,
        output_dir=args.output_dir,
        audio_format=args.audio_format,
    )

    with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:
        # Enumerate chunks to pass index to worker. Each index is the same as the shard_offset.
        chunk_iterator = enumerate(chunked_iterator(manifest_iterable, args.processing_chunk_size))
        futures = {
            executor.submit(worker_func, chunk_with_idx): chunk_with_idx[0] for chunk_with_idx in chunk_iterator
        }
        num_chunks = len(futures)

        logging.info(f"Submitted {num_chunks} chunks to workers.")

        for future in tqdm(as_completed(futures), total=num_chunks, desc="Processing Chunks"):
            chunk_idx = futures[future]
            try:
                result = future.result()
                total_processed_count += result["processed"]
                total_initial_errors += result["initial_errors"]
                total_audio_load_errors += result["audio_load_errors"]
                total_write_errors += result["write_errors"]
                logging.debug(f"Chunk {chunk_idx} finished with result: {result}")
            except Exception as e:
                logging.error(f"Chunk {chunk_idx} failed with exception: {e}", exc_info=True)
                # Increment error count based on chunk size. Difficult to know precisely. Assume all failed.
                total_initial_errors += args.processing_chunk_size

    logging.info("=" * 30 + " Processing Summary " + "=" * 30)
    logging.info(f"Total chunks processed: {num_chunks}")
    logging.info(f"Successfully processed and wrote data for approximately {total_processed_count} entries.")
    total_errors = total_initial_errors + total_audio_load_errors + total_write_errors
    if total_errors > 0:
        logging.warning(f"Encountered errors/skips in {total_errors} potential entries:")
        logging.warning(f"  - Initial processing errors/skips: {total_initial_errors}")
        logging.warning(f"  - Audio loading errors/skips (affecting writes): {total_audio_load_errors}")
        logging.warning(f"  - Writing errors: {total_write_errors}")
        logging.warning("Check logs above (use DEBUG level for more details) for specific entry issues.")
    else:
        logging.info("No significant errors reported.")
    logging.info("Manifest creation finished.")


if __name__ == "__main__":
    main()
