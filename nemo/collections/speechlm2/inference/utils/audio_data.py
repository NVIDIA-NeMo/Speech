# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Audio data loading and output serialization for S2S inference scripts."""

import json
import os

import soundfile as sf

from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
from nemo.collections.speechlm2.inference.streaming.state.s2s_streaming_output import S2SStreamingOutput


def prepare_audio_data(
    audio_file: str,
    default_system_prompt: str | None = None,
    sort_by_duration: bool = True,
) -> tuple[list[str], list[S2SRequestOptions], list[str | None]]:
    """Load audio filepaths and per-stream options from a folder, single file, or manifest.

    ``audio_file`` may point to a single ``.wav`` file, a directory of ``.wav``
    files, or a line-delimited ``.json``/``.jsonl`` inference manifest.  Each
    manifest line must contain ``audio_filepath`` and may also contain
    ``system_prompt`` and ``text``::

        {"audio_filepath": "clip.wav", "text": "...", "system_prompt": "..."}

    If ``system_prompt`` is absent on a line, *default_system_prompt* is used.
    The ``text`` field is returned as optional reference transcript for WER on
    the ASR/user side.

    Returns:
        ``(filepaths, options, ground_truths)`` -- parallel lists of audio paths,
        per-stream request options, and ground-truth texts (``None`` when unavailable).
    """
    audio_file = audio_file.strip()
    if not os.path.isabs(audio_file):
        audio_file = os.path.abspath(audio_file)

    options: list[S2SRequestOptions] = []
    ground_truths: list[str | None] = []

    if os.path.isdir(audio_file):
        filepaths = [os.path.join(audio_file, x) for x in os.listdir(audio_file) if x.endswith(".wav")]
        options = [S2SRequestOptions(system_prompt=default_system_prompt) for _ in filepaths]
        ground_truths = [None] * len(filepaths)
    elif audio_file.endswith(".wav"):
        filepaths = [audio_file]
        options = [S2SRequestOptions(system_prompt=default_system_prompt)]
        ground_truths = [None]
    elif audio_file.endswith((".json", ".jsonl")):
        samples = []
        with open(audio_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    samples.append(json.loads(line))
        filepaths = [get_full_path(entry["audio_filepath"], audio_file) for entry in samples]
        options = [
            S2SRequestOptions(
                system_prompt=entry.get("system_prompt", default_system_prompt),
            )
            for entry in samples
        ]
        ground_truths = [entry.get("text", None) for entry in samples]
    else:
        raise ValueError(f"audio_file `{audio_file}` needs to be a folder, audio file, or manifest file")

    if sort_by_duration:
        durations = [sf.SoundFile(fp).frames for fp in filepaths]
        order = sorted(range(len(filepaths)), key=lambda i: durations[i], reverse=True)
        filepaths = [filepaths[i] for i in order]
        options = [options[i] for i in order]
        ground_truths = [ground_truths[i] for i in order]

    return filepaths, options, ground_truths


def calculate_durations_incl_padding(
    audio_filepaths: list[str],
    pad_audio_to_sec: float | None = None,
    pad_silence_ratio: float | None = None,
    pad_audio_by_sec: float | None = None,
) -> list[float]:
    """Return per-file durations in seconds, accounting for silence padding.

    At most one padding argument may be set; when none are set this
    returns the raw audio durations.
    """
    if sum(x is not None for x in [pad_audio_to_sec, pad_silence_ratio, pad_audio_by_sec]) > 1:
        raise ValueError("Set at most one of: pad_audio_to_sec, pad_silence_ratio, pad_audio_by_sec")
    durations = []
    for fp in audio_filepaths:
        sound = sf.SoundFile(fp)
        dur = sound.frames / sound.samplerate
        if pad_audio_to_sec is not None:
            dur = max(dur, pad_audio_to_sec)
        elif pad_silence_ratio is not None:
            dur *= 1 + pad_silence_ratio
        elif pad_audio_by_sec is not None:
            dur += pad_audio_by_sec
        durations.append(dur)
    return durations


def dump_output_json(
    audio_filepaths: list[str],
    outputs: list[S2SStreamingOutput],
    output_dir: str,
    options: list[S2SRequestOptions],
    ground_truths: list[str | None],
) -> None:
    """Dump inference results to output_processed.json and output_raw.json.

    ``output_processed.json`` strips pad and BOS/EOS tokens and annotates
    turn boundaries with timestamp annotations:

    * ``<|t|>`` -- turn start (BOS position, in seconds)
    * ``<$t$>`` -- turn end   (EOS position, in seconds)

    ``output_raw.json`` preserves the full token stream as-is, including:

    * Pad tokens (e.g. ``<SPECIAL_12>``) for frames with no text output
    * Agent turn markers: ``<s>`` (BOS) and ``</s>`` (EOS)
    * User turn markers (e.g. ``^`` for user BOS, ``</s>`` for user EOS,
      depending on the checkpoint)

    The raw format is useful for debugging token-level model behavior.
    """
    output_processed_path = os.path.join(output_dir, "output_processed.json")
    output_raw_path = os.path.join(output_dir, "output_raw.json")

    with open(output_processed_path, 'w') as f_proc, open(output_raw_path, 'w') as f_raw:
        for audio_filepath, opts, gt, out in zip(audio_filepaths, options, ground_truths, outputs):
            stem = os.path.splitext(os.path.basename(audio_filepath))[0]
            pred_audio_path = os.path.join(output_dir, "wav", f"{stem}.wav")

            record_processed = {
                "id": stem,
                "target_text": "",
                "pred_audio": pred_audio_path,
                "src_text": gt or "",
                "pred_src_text": out.asr_text_with_timestamps or "",
                "pred_text": out.text_with_timestamps or "",
                "system_prompt": opts.system_prompt or "",
            }
            json.dump(record_processed, f_proc, ensure_ascii=False)
            f_proc.write('\n')
            f_proc.flush()

            record_raw = {
                "id": stem,
                "target_text": "",
                "pred_audio": pred_audio_path,
                "src_text": gt or "",
                "pred_src_text": out.raw_asr_text or "",
                "pred_text": out.raw_text or "",
                "system_prompt": opts.system_prompt or "",
            }
            json.dump(record_raw, f_raw, ensure_ascii=False)
            f_raw.write('\n')
            f_raw.flush()
