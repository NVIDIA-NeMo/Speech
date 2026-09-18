#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Run PEE + TransformerCTC speaker-specific word timestamp extraction.

By default this reads one record from the supplied t-SOT JSONL manifest and
prints a JSON object containing only speaker-specific word timestamps. The
timestamps are relative to the WAV file itself; the manifest's source-recording
offset is intentionally not applied.

The decoder argument accepts either the small standalone decoder artifact
created by scripts/extract_transformer_ctc_decoder.py or the original Lightning
checkpoint containing decoder.* weights.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from omegaconf import OmegaConf


SPEECH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SPEECH_ROOT.parent
DIASR_ROOT = PROJECT_ROOT.parent

if str(SPEECH_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEECH_ROOT))

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.parallel_expert_encoder import (
    MultiSpeakerSOTWordTimestampAligner,
    ParallelExpertEncoderPT,
    TransformerCTCDecoder,
)
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


DEFAULT_PEE = PROJECT_ROOT / "pretrained_checkpoints" / "ParallelExpertEncoder_HR8_s16000.nemo"
DEFAULT_DECODER = (
    PROJECT_ROOT
    / "pretrained_checkpoints"
    / "transformer_ctc_decoder_from_Nemotron3_CTC_Timestamp_Adapter_valwer0.1524_epoch10.pt"
)
DEFAULT_TOKENIZER = (
    DIASR_ROOT / "SoundFormer" / "data" / "tokenizer" / "nemotron_3p5_asr_streaming_0p6b" / "tokenizer.model"
)


def parse_record_indices(value: str) -> list[int]:
    """Parse explicit zero-based manifest indices, preserving requested order."""
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("--record-indices must not be empty.")

    indices: list[int] = []
    seen: set[int] = set()
    for component in value.split(","):
        component = component.strip()
        if not component:
            raise argparse.ArgumentTypeError("--record-indices cannot contain an empty component.")
        if "-" in component:
            start_text, separator, end_text = component.partition("-")
            if not separator or "-" in end_text or not start_text.isdecimal() or not end_text.isdecimal():
                raise argparse.ArgumentTypeError(
                    f"Invalid --record-indices range {component!r}; expected a non-negative range such as 3-7."
                )
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Invalid --record-indices range {component!r}; range end must not precede its start."
                )
            component_indices = range(start, end + 1)
        elif component.isdecimal():
            component_indices = (int(component),)
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid --record-indices item {component!r}; expected a non-negative integer or range."
            )

        for index in component_indices:
            if index in seen:
                raise argparse.ArgumentTypeError(f"--record-indices contains duplicate record index {index}.")
            seen.add(index)
            indices.append(index)
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Manifest providing audio_filepath and generation. Cannot be combined with --audio-file/--transcript.",
    )
    record_selection = parser.add_mutually_exclusive_group()
    record_selection.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="Zero-based manifest record to run (default: 0).",
    )
    record_selection.add_argument(
        "--record-indices",
        type=parse_record_indices,
        default=None,
        metavar="INDICES",
        help=(
            "Comma-separated manifest record indices and inclusive ranges, for example '0,3-7'. "
            "Records are processed in this order."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=("Maximum number of selected manifest records to run in one padded PEE/CTC/DP batch " "(default: 1)."),
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="Optional WAV file override. Requires --transcript.",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help="Optional t-SOT transcript override, for example '<spk:0> hello <spk:1> yes'.",
    )
    parser.add_argument(
        "--sot-field",
        default="generation",
        help="Manifest field containing Nemotron-Transcribe t-SOT output (default: generation).",
    )
    parser.add_argument("--pee-nemo", type=Path, default=DEFAULT_PEE, help=f"PEE archive (default: {DEFAULT_PEE}).")
    parser.add_argument(
        "--ctc-checkpoint",
        type=Path,
        default=DEFAULT_DECODER,
        help=f"Standalone decoder artifact or Lightning checkpoint (default: {DEFAULT_DECODER}).",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help=f"Matching SentencePiece model (default: {DEFAULT_TOKENIZER}).",
    )
    parser.add_argument(
        "--speaker-logprob-weight",
        type=float,
        default=0.25,
        help=(
            "Weight for the Sortformer log-sigmoid prior in CTC DP "
            "(default: 0.25; 1.0 multiplies by max(activity, epsilon))."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    parser.add_argument(
        "--model-dtype",
        choices=("bf16", "fp32"),
        default="bf16" if torch.cuda.is_available() else "fp32",
        help="PEE/CTC dtype; CPU requires fp32.",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="Optional seconds to add to output. Keep zero for chunk-local WAV timestamps.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional destination JSON path. With multiple manifest records this is JSONL; "
            "otherwise it is one JSON object. Without it, JSON is written to stdout."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for per-record json/, ctm/, seglst/, and rttm/ files. "
            "Required to write CTM, SegLST, or RTTM for multiple manifest records."
        ),
    )
    parser.add_argument(
        "--output-ctm",
        type=Path,
        default=None,
        help=(
            "Optional Gecko-compatible recording-level CTM path. CTM column 1 encodes "
            "the t-SOT speaker and a global segment ID; column 2 is channel 1."
        ),
    )
    parser.add_argument(
        "--output-seglst",
        type=Path,
        default=None,
        help="Optional merged speaker-segment JSON (.seglst) path.",
    )
    parser.add_argument(
        "--output-rttm",
        type=Path,
        default=None,
        help="Optional merged speaker-segment RTTM path.",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.1,
        help=(
            "Merge adjacent words in the same speaker/t-SOT turn when their gap is at most this "
            "many seconds (default: 0.1). This controls SegLST and Gecko CTM segments; use -1 "
            "for one segment per word."
        ),
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Optional session ID for SegLST, RTTM, and output filenames. Defaults to the input WAV filename "
            "without its extension."
        ),
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _manifest_record_from_mapping(record: Any, record_index: int, sot_field: str) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Manifest record {record_index} must be a JSON object.")
    audio = record.get("audio_filepath")
    transcript = record.get(sot_field)
    if not isinstance(audio, str) or not audio:
        raise ValueError(f"Manifest record {record_index} has no non-empty audio_filepath.")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError(f"Manifest record {record_index} has no non-empty {sot_field!r} string.")
    return require_file(Path(audio), "Manifest audio file"), transcript


def load_manifest_records(path: Path, record_indices: list[int], sot_field: str) -> list[tuple[int, Path, str]]:
    """Load explicit manifest rows once, preserving the user-requested order."""
    if not record_indices:
        raise ValueError("At least one manifest record index is required.")
    if any(index < 0 for index in record_indices):
        raise ValueError("Manifest record indices must be non-negative.")
    if len(record_indices) != len(set(record_indices)):
        raise ValueError("Manifest record indices must be unique.")

    remaining = set(record_indices)
    selected: dict[int, tuple[Path, str]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for index, line in enumerate(input_file):
            if index not in remaining:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Manifest record {index} is not valid JSON: {path}") from error
            selected[index] = _manifest_record_from_mapping(record, index, sot_field)
            remaining.remove(index)
            if not remaining:
                break

    if remaining:
        unavailable = ", ".join(str(index) for index in sorted(remaining))
        raise IndexError(f"Manifest has no record(s) at index {unavailable}: {path}")
    return [(index, *selected[index]) for index in record_indices]


def load_manifest_record(path: Path, record_index: int, sot_field: str) -> tuple[Path, str]:
    """Backward-compatible single-record manifest loader."""
    _, audio_path, transcript = load_manifest_records(path, [record_index], sot_field)[0]
    return audio_path, transcript


def _to_plain_mapping(config: Any, label: str) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{label} must be a mapping, got {type(config).__name__}.")
    return dict(config)


def _load_decoder_payload(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"CTC checkpoint must be a mapping, got {type(payload).__name__}.")

    if payload.get("format") == "transformer_ctc_decoder_state_dict_v1":
        config = _to_plain_mapping(payload.get("decoder_config"), "decoder_config")
        state_dict = payload.get("state_dict")
    else:
        hyper_parameters = payload.get("hyper_parameters")
        if not isinstance(hyper_parameters, Mapping):
            raise RuntimeError(
                "Expected either a standalone TransformerCTC decoder artifact or a Lightning checkpoint "
                "with hyper_parameters.cfg.decoder."
            )
        model_config = hyper_parameters.get("cfg")
        if OmegaConf.is_config(model_config):
            decoder_config = model_config.get("decoder")
        elif isinstance(model_config, Mapping):
            decoder_config = model_config.get("decoder")
        else:
            decoder_config = getattr(model_config, "decoder", None)
        config = _to_plain_mapping(decoder_config, "Lightning cfg.decoder")
        source_state = payload.get("state_dict")
        if not isinstance(source_state, Mapping):
            raise RuntimeError("Lightning checkpoint has no mapping-valued state_dict.")
        state_dict = {
            key[len("decoder.") :]: value
            for key, value in source_state.items()
            if isinstance(key, str) and key.startswith("decoder.")
        }

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("No decoder state_dict tensors were found in the checkpoint.")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise RuntimeError("Decoder state_dict contains a non-tensor value.")
    return config, dict(state_dict)


def load_decoder(path: Path, device: torch.device, dtype: torch.dtype) -> tuple[TransformerCTCDecoder, dict[str, Any]]:
    decoder_config, decoder_state = _load_decoder_payload(path)
    accepted = set(inspect.signature(TransformerCTCDecoder.__init__).parameters)
    accepted.discard("self")
    init_config = {key: value for key, value in decoder_config.items() if key in accepted}
    decoder = TransformerCTCDecoder(**init_config)
    decoder.load_state_dict(decoder_state, strict=True)
    return decoder.to(device=device, dtype=dtype).eval(), decoder_config


def validate_tokenizer(tokenizer: SentencePieceTokenizer, decoder_config: Mapping[str, Any]) -> None:
    vocabulary = decoder_config.get("vocabulary")
    if vocabulary is None:
        return
    vocabulary = list(vocabulary)
    if int(tokenizer.vocab_size) != len(vocabulary):
        raise ValueError(
            "Tokenizer vocabulary size does not match the decoder configuration: "
            f"{tokenizer.vocab_size} != {len(vocabulary)}."
        )
    actual_vocabulary = tokenizer.ids_to_tokens(list(range(len(vocabulary))))
    mismatch = next(
        (index for index, (actual, expected) in enumerate(zip(actual_vocabulary, vocabulary)) if actual != expected),
        None,
    )
    if mismatch is not None:
        raise ValueError(
            "Tokenizer pieces do not match the decoder vocabulary at ID "
            f"{mismatch}: {actual_vocabulary[mismatch]!r} != {vocabulary[mismatch]!r}."
        )


def build_preprocessor(device: torch.device) -> AudioToMelSpectrogramPreprocessor:
    # PEE normalizes ASR and Sortformer mel features internally, so the shared
    # front-end deliberately returns unnormalized log-mels.
    return (
        AudioToMelSpectrogramPreprocessor(
            sample_rate=16000,
            normalize=None,
            window_size=0.025,
            window_stride=0.01,
            window="hann",
            features=128,
            n_fft=512,
            log=True,
            frame_splicing=1,
            dither=0.0,
            pad_to=0,
            pad_value=0.0,
        )
        .to(device)
        .eval()
    )


def _load_waveform_samples(path: Path) -> tuple[torch.Tensor, float]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio for PEE, got {sample_rate} Hz: {path}")
    waveform = torch.from_numpy(samples.mean(axis=1)).contiguous()
    if waveform.numel() == 0:
        raise ValueError(f"Audio file contains no samples: {path}")
    return waveform, waveform.numel() / float(sample_rate)


def load_waveforms(paths: list[Path], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Read, mono-mix, and right-pad several 16 kHz recordings for one model batch."""
    if not paths:
        raise ValueError("At least one audio path is required.")

    loaded = [_load_waveform_samples(path) for path in paths]
    lengths_cpu = torch.tensor([waveform.numel() for waveform, _ in loaded], dtype=torch.long)
    max_samples = int(lengths_cpu.max().item())
    waveforms_cpu = torch.zeros((len(loaded), max_samples), dtype=torch.float32)
    for index, (waveform, _) in enumerate(loaded):
        waveforms_cpu[index, : waveform.numel()] = waveform
    durations = [duration for _, duration in loaded]
    return waveforms_cpu.to(device=device), lengths_cpu.to(device=device), durations


def load_waveform(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Backward-compatible one-record wrapper around :func:`load_waveforms`."""
    waveform, length, durations = load_waveforms([path], device)
    return waveform, length, durations[0]


def json_result(
    audio_path: Path,
    duration: float,
    result: Mapping[str, Any],
    *,
    manifest_record_index: int | None = None,
) -> dict[str, Any]:
    output = {
        "audio_filepath": str(audio_path),
        "duration": duration,
        "speaker_word_timestamps": result["speaker_word_timestamps"],
        "speaker_tag_to_sortformer_column": result["speaker_tag_to_sortformer_column"],
        "alignment": {
            "mode": result["alignment_mode"],
            "requested_mode": result.get("requested_alignment_mode", result["alignment_mode"]),
            "speaker_assignment_mode": result["speaker_assignment_mode"],
            "ctc_frame_seconds": result["ctc_frame_seconds"],
            "sortformer_frame_seconds": result["sortformer_frame_seconds"],
            "num_ctc_frames": result["num_ctc_frames"],
            "num_sortformer_frames": result["num_sortformer_frames"],
            "diagnostics": result["alignment_diagnostics"],
        },
    }
    if manifest_record_index is not None:
        output["manifest_record_index"] = manifest_record_index
    return output


def _format_time(value: float) -> str:
    """Format a finite timestamp consistently with the multi-speaker aligner."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Timestamp must be finite, got {value!r}.")
    if value == 0.0:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _coerce_speaker_tag(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("speaker_tag must be an integer, not a boolean.")
    try:
        speaker_tag = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"speaker_tag must be an integer, got {value!r}.") from error
    if speaker_tag < 0:
        raise ValueError(f"speaker_tag must be non-negative, got {speaker_tag}.")
    return speaker_tag


def _time_interval(start_value: Any, end_value: Any, description: str) -> tuple[float, float]:
    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid interval for {description}.") from error
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError(f"Invalid interval [{start_value!r}, {end_value!r}] for {description}.")
    return start, end


def _session_id(audio_path: Path, requested_session_id: str | None) -> str:
    session_id = requested_session_id.strip() if requested_session_id else audio_path.stem
    if not session_id or len(session_id.split()) != 1:
        raise ValueError("Session ID must be one non-empty token; use --session-id to override the WAV stem.")
    return session_id


def _normalized_timestamp_words(speaker_word_timestamps: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and flatten aligned words without changing their speaker-tag semantics."""
    if not isinstance(speaker_word_timestamps, Mapping):
        raise TypeError("speaker_word_timestamps must be a mapping.")

    normalized_words: list[dict[str, Any]] = []
    for timestamp_key, word_rows in speaker_word_timestamps.items():
        if not isinstance(word_rows, list):
            raise TypeError("speaker_word_timestamps values must be lists.")
        for row_index, word_row in enumerate(word_rows):
            if not isinstance(word_row, Mapping):
                raise TypeError("Each speaker word timestamp must be a mapping.")
            speaker_tag = _coerce_speaker_tag(word_row.get("speaker_tag", timestamp_key))
            word = str(word_row.get("word", "")).strip()
            if not word:
                raise ValueError("Cannot write an empty aligned word.")
            start, end = _time_interval(word_row.get("start"), word_row.get("end"), f"word {word!r}")
            word_index = word_row.get("word_index", row_index)
            try:
                word_index = int(word_index)
            except (TypeError, ValueError) as error:
                raise ValueError(f"word_index must be an integer for {word!r}.") from error
            turn_index = word_row.get("turn_index")
            if turn_index is not None:
                try:
                    turn_index = int(turn_index)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"turn_index must be an integer or null for {word!r}.") from error
            normalized_words.append(
                {
                    "word": word,
                    "start": start,
                    "end": end,
                    "speaker_tag": speaker_tag,
                    "word_index": word_index,
                    "turn_index": turn_index,
                    "sortformer_column": word_row.get("sortformer_column"),
                }
            )
    return normalized_words


def _merged_timestamp_word_segments(
    speaker_word_timestamps: Mapping[str, Any], merge_threshold: float
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Group timestamp words for the shared SegLST and Gecko-CTM segmentation.

    The t-SOT ``turn_index`` is a hard boundary even when a reappearing speaker's
    words are temporally adjacent. This guarantees that the two formats expose
    exactly the same speaker-turn segmentation.
    """
    if not math.isfinite(merge_threshold) or merge_threshold < -1.0:
        raise ValueError("--merge-threshold must be finite and at least -1 seconds.")

    grouped_words: dict[int, list[dict[str, Any]]] = {}
    for word_row in _normalized_timestamp_words(speaker_word_timestamps):
        grouped_words.setdefault(word_row["speaker_tag"], []).append(word_row)

    merged_segments: list[tuple[int, list[dict[str, Any]]]] = []
    for speaker_tag, words in grouped_words.items():
        words.sort(key=lambda word: (word["start"], word["end"], word["word_index"]))
        current_words: list[dict[str, Any]] = []
        current_end: float | None = None
        for word in words:
            if current_words and (
                word["turn_index"] != current_words[-1]["turn_index"]
                or merge_threshold < 0.0
                or word["start"] - current_end > merge_threshold
            ):
                merged_segments.append((speaker_tag, current_words))
                current_words = []
                current_end = None
            current_words.append(word)
            current_end = word["end"] if current_end is None else max(current_end, word["end"])
        if current_words:
            merged_segments.append((speaker_tag, current_words))

    return sorted(
        merged_segments,
        key=lambda segment: (
            segment[1][0]["start"],
            max(word["end"] for word in segment[1]),
            segment[0],
            segment[1][0]["word_index"],
        ),
    )


def build_ctm_lines(speaker_word_timestamps: Mapping[str, Any], merge_threshold: float) -> list[str]:
    """Build Gecko-compatible CTM rows from merged speaker/t-SOT segments.

    Gecko overloads CTM column one as ``spkN_<segment-id>_audio`` and ignores
    column two when assigning speakers. Segment IDs must therefore be globally
    unique across speakers. Gecko groups only adjacent, time-sorted rows with
    the same ID, so an overlapping merged segment is split into separate IDs
    whenever another segment's word interrupts it on the time axis.
    """
    records: list[tuple[float, float, int, int, int, str]] = []
    for semantic_segment_id, (speaker_tag, words) in enumerate(
        _merged_timestamp_word_segments(speaker_word_timestamps, merge_threshold)
    ):
        for word_row in words:
            word = word_row["word"]
            if len(word.split()) != 1:
                raise ValueError(f"CTM words must be one token, got {word!r}.")
            records.append(
                (
                    word_row["start"],
                    word_row["end"],
                    speaker_tag,
                    semantic_segment_id,
                    word_row["word_index"],
                    word,
                )
            )
    records.sort(key=lambda record: record[:-1])

    lines: list[str] = []
    previous_semantic_segment_id: int | None = None
    gecko_segment_id = -1
    for start, end, speaker_tag, semantic_segment_id, _, word in records:
        if semantic_segment_id != previous_semantic_segment_id:
            gecko_segment_id += 1
            previous_semantic_segment_id = semantic_segment_id
        gecko_utterance_id = f"spk{speaker_tag}_{gecko_segment_id:05d}_audio"
        lines.append(f"{gecko_utterance_id} 1 {start:.2f} {end - start:.2f} {word} -1.00")
    return lines


def _segment_from_words(
    session_id: str, audio_path: Path, speaker_tag: int, words: list[dict[str, Any]]
) -> dict[str, Any]:
    start = words[0]["start"]
    end = max(word["end"] for word in words)
    columns = {word["sortformer_column"] for word in words}
    turn_indices = {word["turn_index"] for word in words}
    return {
        "end_time": _format_time(end),
        "session_id": session_id,
        "speaker": f"spk:{speaker_tag}",
        "start_time": _format_time(start),
        "words": " ".join(word["word"] for word in words),
        "speaker_tag": speaker_tag,
        "turn_index": turn_indices.pop() if len(turn_indices) == 1 else None,
        "sortformer_column": columns.pop() if len(columns) == 1 else None,
        "audio_filepath": str(audio_path),
        "num_words": len(words),
    }


def build_seglst_segments(
    session_id: str,
    audio_path: Path,
    speaker_word_timestamps: Mapping[str, Any],
    merge_threshold: float,
) -> list[dict[str, Any]]:
    """Merge same-speaker word timestamps without crossing original t-SOT turns."""
    segments = [
        _segment_from_words(session_id, audio_path, speaker_tag, words)
        for speaker_tag, words in _merged_timestamp_word_segments(speaker_word_timestamps, merge_threshold)
    ]

    return sorted(
        segments,
        key=lambda segment: (
            segment["session_id"],
            float(segment["start_time"]),
            float(segment["end_time"]),
            segment["speaker_tag"],
        ),
    )


def build_rttm_lines(
    session_id: str,
    speaker_word_timestamps: Mapping[str, Any],
    merge_threshold: float,
) -> list[str]:
    """Build standard RTTM speaker segments from merged timestamp words.

    RTTM uses the same t-SOT turn boundaries and ``merge_threshold`` as
    SegLST and CTM. The recording ID is ``session_id`` and channel is 1.
    """
    lines: list[str] = []
    for speaker_tag, words in _merged_timestamp_word_segments(speaker_word_timestamps, merge_threshold):
        start = words[0]["start"]
        end = max(word["end"] for word in words)
        lines.append(f"SPEAKER {session_id} 1 {start:.3f} {end - start:.3f} " f"<NA> <NA> spk:{speaker_tag} <NA> <NA>")
    return lines


def _write_text(path: Path, contents: str) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    os.replace(temporary_path, path)
    return path


def write_ctm(path: Path, lines: list[str]) -> Path:
    return _write_text(path, "".join(line + "\n" for line in lines))


def write_seglst(path: Path, segments: list[dict[str, Any]]) -> Path:
    return _write_text(path, json.dumps(segments, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rttm(path: Path, lines: list[str]) -> Path:
    return _write_text(path, "".join(line + "\n" for line in lines))


def _validate_output_paths(*paths: Path | None) -> None:
    resolved_paths = [path.expanduser().resolve() for path in paths if path is not None]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError(
            "--output-json, --output-ctm, --output-seglst, and --output-rttm must refer to different paths."
        )


def _record_output_stem(session_id: str, manifest_record_index: int | None) -> str:
    safe_session_id = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_" for character in session_id
    ).strip(".")
    if not safe_session_id:
        safe_session_id = "session"
    prefix = "" if manifest_record_index is None else f"{manifest_record_index:06d}_"
    return f"{prefix}{safe_session_id}"


def write_record_output_dir(
    output_dir: Path,
    *,
    session_id: str,
    manifest_record_index: int | None,
    audio_path: Path,
    result: Mapping[str, Any],
    output: Mapping[str, Any],
    merge_threshold: float,
) -> dict[str, Path]:
    """Write collision-safe per-record JSON, CTM, SegLST, and RTTM artifacts."""
    root = output_dir.expanduser().resolve()
    stem = _record_output_stem(session_id, manifest_record_index)
    json_path = _write_text(root / "json" / f"{stem}.json", json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    ctm_path = write_ctm(
        root / "ctm" / f"{stem}.ctm",
        build_ctm_lines(result["speaker_word_timestamps"], merge_threshold),
    )
    seglst_path = write_seglst(
        root / "seglst" / f"{stem}.seglst",
        build_seglst_segments(session_id, audio_path, result["speaker_word_timestamps"], merge_threshold),
    )
    rttm_path = write_rttm(
        root / "rttm" / f"{stem}.rttm",
        build_rttm_lines(session_id, result["speaker_word_timestamps"], merge_threshold),
    )
    return {"json": json_path, "ctm": ctm_path, "seglst": seglst_path, "rttm": rttm_path}


def main() -> int:
    args = parse_args()
    _validate_output_paths(args.output_json, args.output_ctm, args.output_seglst, args.output_rttm)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    pee_path = require_file(args.pee_nemo, "PEE .nemo archive")
    decoder_path = require_file(args.ctc_checkpoint, "TransformerCTC decoder checkpoint")
    tokenizer_path = require_file(args.tokenizer_model, "SentencePiece tokenizer model")

    has_manifest_input = args.input_jsonl is not None
    has_direct_input = args.audio_file is not None or args.transcript is not None
    if has_manifest_input and has_direct_input:
        raise ValueError("Choose either --input-jsonl or the --audio-file/--transcript pair, not both.")
    if has_direct_input:
        if args.audio_file is None or args.transcript is None:
            raise ValueError("--audio-file and --transcript must be supplied together.")
        if args.record_indices is not None:
            raise ValueError("--record-indices requires --input-jsonl.")
        if args.batch_size != 1:
            raise ValueError("--batch-size greater than one requires --input-jsonl with multiple records.")
        records: list[tuple[int | None, Path, str]] = [
            (None, require_file(args.audio_file, "--audio-file"), args.transcript)
        ]
    elif has_manifest_input:
        manifest_path = require_file(args.input_jsonl, "Input manifest")
        record_indices = args.record_indices if args.record_indices is not None else [args.record_index]
        records = load_manifest_records(manifest_path, record_indices, args.sot_field)
    else:
        raise ValueError("Provide either --input-jsonl or both --audio-file and --transcript.")

    multiple_records = len(records) > 1
    if multiple_records and (
        args.output_ctm is not None or args.output_seglst is not None or args.output_rttm is not None
    ):
        raise ValueError(
            "--output-ctm, --output-seglst, and --output-rttm accept one recording only; "
            "use --output-dir for per-record files."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device={device} was requested, but CUDA is unavailable.")
    if device.type == "cpu" and args.model_dtype != "fp32":
        raise ValueError("CPU inference requires --model-dtype fp32.")
    dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32

    print(f"Loading PEE: {pee_path}", file=sys.stderr, flush=True)
    pee = ParallelExpertEncoderPT.load_from_nemo(str(pee_path), map_location="cpu", strict=True)
    pee = pee.to(device=device, dtype=dtype).eval()

    print(f"Loading TransformerCTC decoder: {decoder_path}", file=sys.stderr, flush=True)
    decoder, decoder_config = load_decoder(decoder_path, device, dtype)
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    validate_tokenizer(tokenizer, decoder_config)
    preprocessor = build_preprocessor(device)

    extractor = MultiSpeakerSOTWordTimestampAligner(
        encoder=pee,
        ctc_decoder=decoder,
        tokenizer=tokenizer,
        speaker_logprob_weight=args.speaker_logprob_weight,
    )

    completed: list[tuple[int | None, Path, float, str, Mapping[str, Any], dict[str, Any]]] = []
    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start : batch_start + args.batch_size]
        batch_indices = [record_index for record_index, _, _ in batch_records]
        print(
            f"Running timestamp extraction batch of {len(batch_records)} record(s): {batch_indices}",
            file=sys.stderr,
            flush=True,
        )
        if len(batch_records) == 1:
            record_index, audio_path, transcript = batch_records[0]
            waveform, waveform_length, duration = load_waveform(audio_path, device)
            batch_results: list[Mapping[str, Any]] = [
                extractor.extract_from_audio(
                    input_signal=waveform,
                    input_signal_length=waveform_length,
                    preprocessor=preprocessor,
                    sot_transcript=transcript,
                    audio_duration=duration,
                    time_offset=args.time_offset,
                )
            ]
            durations = [duration]
        else:
            audio_paths = [audio_path for _, audio_path, _ in batch_records]
            waveforms, waveform_lengths, durations = load_waveforms(audio_paths, device)
            batch_results = extractor.extract_from_audio_batch(
                input_signal=waveforms,
                input_signal_length=waveform_lengths,
                preprocessor=preprocessor,
                sot_transcripts=[transcript for _, _, transcript in batch_records],
                audio_durations=durations,
                time_offsets=[args.time_offset] * len(batch_records),
            )
            if len(batch_results) != len(batch_records):
                raise RuntimeError(
                    "extract_from_audio_batch returned an unexpected number of results: "
                    f"{len(batch_results)} != {len(batch_records)}."
                )

        for (record_index, audio_path, _), duration, result in zip(batch_records, durations, batch_results):
            session_id = _session_id(audio_path, args.session_id)
            output = json_result(
                audio_path,
                duration,
                result,
                manifest_record_index=record_index if multiple_records else None,
            )
            completed.append((record_index, audio_path, duration, session_id, result, output))
            if args.output_dir is not None:
                written = write_record_output_dir(
                    args.output_dir,
                    session_id=session_id,
                    manifest_record_index=record_index,
                    audio_path=audio_path,
                    result=result,
                    output=output,
                    merge_threshold=args.merge_threshold,
                )
                print(
                    "Wrote per-record outputs: "
                    f"{written['json']}, {written['ctm']}, {written['seglst']}, {written['rttm']}",
                    file=sys.stderr,
                    flush=True,
                )

    if len(completed) != len(records):
        raise RuntimeError("Not every selected manifest record produced an extraction result.")

    if len(completed) == 1:
        _, audio_path, _, session_id, result, _ = completed[0]
        if args.output_ctm is not None:
            ctm_path = write_ctm(
                args.output_ctm,
                build_ctm_lines(result["speaker_word_timestamps"], args.merge_threshold),
            )
            print(f"Wrote CTM: {ctm_path}", file=sys.stderr, flush=True)
        if args.output_seglst is not None:
            seglst_path = write_seglst(
                args.output_seglst,
                build_seglst_segments(
                    session_id,
                    audio_path,
                    result["speaker_word_timestamps"],
                    args.merge_threshold,
                ),
            )
            print(f"Wrote SegLST: {seglst_path}", file=sys.stderr, flush=True)
        if args.output_rttm is not None:
            rttm_path = write_rttm(
                args.output_rttm,
                build_rttm_lines(session_id, result["speaker_word_timestamps"], args.merge_threshold),
            )
            print(f"Wrote RTTM: {rttm_path}", file=sys.stderr, flush=True)

    outputs = [output for _, _, _, _, _, output in completed]
    if len(outputs) == 1:
        rendered = json.dumps(outputs[0], ensure_ascii=False, indent=2, allow_nan=False)
    else:
        rendered = "\n".join(json.dumps(output, ensure_ascii=False, allow_nan=False) for output in outputs)

    if args.output_json is None:
        print(rendered)
    else:
        output_path = _write_text(args.output_json, rendered + "\n")
        output_kind = "JSONL" if multiple_records else "timestamp JSON"
        print(f"Wrote {output_kind}: {output_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
