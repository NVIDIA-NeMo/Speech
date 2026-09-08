#!/usr/bin/env python3
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
    PEETransformerCTCTimestampExtractor,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Manifest providing audio_filepath and generation. Cannot be combined with --audio-file/--transcript.",
    )
    parser.add_argument("--record-index", type=int, default=0, help="Zero-based manifest record to run.")
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
        "--alignment-mode",
        choices=("parallel", "serialized"),
        default="serialized",
        help="Follow the t-SOT word order on one CTC path (default: serialized).",
    )
    parser.add_argument(
        "--speaker-assignment-mode",
        choices=("optimal", "identity"),
        default="optimal",
        help="Map t-SOT speaker tags to Sortformer columns (default: optimal).",
    )
    parser.add_argument(
        "--speaker-logprob-weight",
        type=float,
        default=0.25,
        help="Weight for the Sortformer log-sigmoid prior in CTC DP (default: 0.25; 1.0 multiplies by max(activity, epsilon)).",
    )
    parser.add_argument(
        "--parallel-speaker-gate-threshold",
        type=float,
        default=0.5,
        help=(
            "Hard-gate non-blank CTC emissions below this Sortformer probability in parallel mode "
            "(default: 0.5). Use a negative value to disable the hard gate."
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
        help="Optional destination JSON path. Without it, JSON is written to stdout.",
    )
    parser.add_argument(
        "--output-ctm",
        type=Path,
        default=None,
        help=(
            "Optional recording-level CTM path. All speakers are written to one file; "
            "CTM column 2 is the numeric t-SOT speaker tag."
        ),
    )
    parser.add_argument(
        "--output-seglst",
        type=Path,
        default=None,
        help="Optional merged speaker-segment JSON (.seglst) path.",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.1,
        help=(
            "Merge adjacent words only when they have the same speaker and their gap is at most "
            "this many seconds (default: 0.1). Use -1 to write one segment per word."
        ),
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session ID for CTM/seglst. Defaults to the input WAV filename without its extension.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def load_manifest_record(path: Path, record_index: int, sot_field: str) -> tuple[Path, str]:
    if record_index < 0:
        raise ValueError("--record-index must be non-negative.")
    with path.open("r", encoding="utf-8") as input_file:
        for index, line in enumerate(input_file):
            if index != record_index:
                continue
            record = json.loads(line)
            audio = record.get("audio_filepath")
            transcript = record.get(sot_field)
            if not isinstance(audio, str) or not audio:
                raise ValueError(f"Manifest record {record_index} has no non-empty audio_filepath.")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(f"Manifest record {record_index} has no non-empty {sot_field!r} string.")
            return require_file(Path(audio), "Manifest audio file"), transcript
    raise IndexError(f"Manifest has no record at --record-index {record_index}: {path}")


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
        (
            index
            for index, (actual, expected) in enumerate(zip(actual_vocabulary, vocabulary))
            if actual != expected
        ),
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
    return AudioToMelSpectrogramPreprocessor(
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
    ).to(device).eval()


def load_waveform(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio for PEE, got {sample_rate} Hz: {path}")
    waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0).to(device=device)
    length = torch.tensor([waveform.shape[-1]], dtype=torch.long, device=device)
    duration = waveform.shape[-1] / float(sample_rate)
    return waveform, length, duration


def json_result(audio_path: Path, duration: float, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audio_filepath": str(audio_path),
        "duration": duration,
        "speaker_word_timestamps": result["speaker_word_timestamps"],
        "speaker_tag_to_sortformer_column": result["speaker_tag_to_sortformer_column"],
        "alignment": {
            "mode": result["alignment_mode"],
            "speaker_assignment_mode": result["speaker_assignment_mode"],
            "ctc_frame_seconds": result["ctc_frame_seconds"],
            "sortformer_frame_seconds": result["sortformer_frame_seconds"],
            "num_ctc_frames": result["num_ctc_frames"],
            "num_sortformer_frames": result["num_sortformer_frames"],
            "diagnostics": result["alignment_diagnostics"],
        },
    }


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
        raise ValueError(
            "CTM/seglst session ID must be one non-empty token; use --session-id to override the WAV stem."
        )
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
            normalized_words.append(
                {
                    "word": word,
                    "start": start,
                    "end": end,
                    "speaker_tag": speaker_tag,
                    "word_index": word_index,
                    "sortformer_column": word_row.get("sortformer_column"),
                }
            )
    return normalized_words


def build_ctm_lines(session_id: str, speaker_word_timestamps: Mapping[str, Any]) -> list[str]:
    """Build one CTM stream with numeric t-SOT speaker IDs in column two."""
    if not session_id or len(session_id.split()) != 1:
        raise ValueError(f"CTM session ID must be one non-empty token, got {session_id!r}.")

    words = _normalized_timestamp_words(speaker_word_timestamps)
    for word_row in words:
        if len(word_row["word"].split()) != 1:
            raise ValueError(f"CTM words must be one token, got {word_row['word']!r}.")
    words.sort(key=lambda word: (word["start"], word["end"], word["speaker_tag"], word["word_index"]))
    return [
        f"{session_id} {word['speaker_tag']} {word['start']:08.2f} "
        f"{word['end'] - word['start']:.2f} {word['word']} -1.00"
        for word in words
    ]


def _segment_from_words(
    session_id: str, audio_path: Path, speaker_tag: int, words: list[dict[str, Any]]
) -> dict[str, Any]:
    start = words[0]["start"]
    end = max(word["end"] for word in words)
    columns = {word["sortformer_column"] for word in words}
    return {
        "end_time": _format_time(end),
        "session_id": session_id,
        "speaker": f"spk:{speaker_tag}",
        "start_time": _format_time(start),
        "words": " ".join(word["word"] for word in words),
        "speaker_tag": speaker_tag,
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
    """Merge same-speaker word timestamps into local-time SegLST segments."""
    if not math.isfinite(merge_threshold) or merge_threshold < -1.0:
        raise ValueError("--merge-threshold must be finite and at least -1 seconds.")

    grouped_words: dict[int, list[dict[str, Any]]] = {}
    for word_row in _normalized_timestamp_words(speaker_word_timestamps):
        grouped_words.setdefault(word_row["speaker_tag"], []).append(word_row)

    segments: list[dict[str, Any]] = []
    for speaker_tag, words in grouped_words.items():
        words.sort(key=lambda word: (word["start"], word["end"], word["word_index"]))
        current_words: list[dict[str, Any]] = []
        current_end: float | None = None
        for word in words:
            if (
                current_words
                and (merge_threshold < 0.0 or word["start"] - current_end > merge_threshold)
            ):
                segments.append(_segment_from_words(session_id, audio_path, speaker_tag, current_words))
                current_words = []
                current_end = None
            current_words.append(word)
            current_end = word["end"] if current_end is None else max(current_end, word["end"])
        if current_words:
            segments.append(_segment_from_words(session_id, audio_path, speaker_tag, current_words))

    return sorted(
        segments,
        key=lambda segment: (
            segment["session_id"],
            float(segment["start_time"]),
            float(segment["end_time"]),
            segment["speaker_tag"],
        ),
    )


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


def _validate_output_paths(*paths: Path | None) -> None:
    resolved_paths = [path.expanduser().resolve() for path in paths if path is not None]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("--output-json, --output-ctm, and --output-seglst must refer to different paths.")


def main() -> int:
    args = parse_args()
    _validate_output_paths(args.output_json, args.output_ctm, args.output_seglst)
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
        audio_path = require_file(args.audio_file, "--audio-file")
        transcript = args.transcript
    elif has_manifest_input:
        manifest_path = require_file(args.input_jsonl, "Input manifest")
        audio_path, transcript = load_manifest_record(manifest_path, args.record_index, args.sot_field)
    else:
        raise ValueError("Provide either --input-jsonl or both --audio-file and --transcript.")
    session_id = _session_id(audio_path, args.session_id)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device={device} was requested, but CUDA is unavailable.")
    if device.type == "cpu" and args.model_dtype != "fp32":
        raise ValueError("CPU inference requires --model-dtype fp32.")
    if args.parallel_speaker_gate_threshold < 0.0:
        parallel_gate_threshold = None
    elif args.parallel_speaker_gate_threshold > 1.0:
        raise ValueError("--parallel-speaker-gate-threshold must be between zero and one, or negative to disable.")
    else:
        parallel_gate_threshold = args.parallel_speaker_gate_threshold
    dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32

    print(f"Loading PEE: {pee_path}", file=sys.stderr, flush=True)
    pee = ParallelExpertEncoderPT.load_from_nemo(str(pee_path), map_location="cpu", strict=True)
    pee = pee.to(device=device, dtype=dtype).eval()

    print(f"Loading TransformerCTC decoder: {decoder_path}", file=sys.stderr, flush=True)
    decoder, decoder_config = load_decoder(decoder_path, device, dtype)
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    validate_tokenizer(tokenizer, decoder_config)
    preprocessor = build_preprocessor(device)

    extractor = PEETransformerCTCTimestampExtractor(
        encoder=pee,
        ctc_decoder=decoder,
        tokenizer=tokenizer,
        alignment_mode=args.alignment_mode,
        speaker_assignment_mode=args.speaker_assignment_mode,
        speaker_logprob_weight=args.speaker_logprob_weight,
        parallel_speaker_gate_threshold=parallel_gate_threshold,
    )

    waveform, waveform_length, duration = load_waveform(audio_path, device)
    print(f"Running timestamp extraction: {audio_path}", file=sys.stderr, flush=True)
    result = extractor.extract_from_audio(
        input_signal=waveform,
        input_signal_length=waveform_length,
        preprocessor=preprocessor,
        sot_transcript=transcript,
        audio_duration=duration,
        time_offset=args.time_offset,
    )
    output = json_result(audio_path, duration, result)
    rendered = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)

    if args.output_ctm is not None:
        ctm_path = write_ctm(args.output_ctm, build_ctm_lines(session_id, result["speaker_word_timestamps"]))
        print(f"Wrote CTM: {ctm_path}", file=sys.stderr, flush=True)
    if args.output_seglst is not None:
        segments = build_seglst_segments(
            session_id,
            audio_path,
            result["speaker_word_timestamps"],
            args.merge_threshold,
        )
        seglst_path = write_seglst(args.output_seglst, segments)
        print(f"Wrote SegLST: {seglst_path}", file=sys.stderr, flush=True)

    if args.output_json is None:
        print(rendered)
    else:
        output_path = _write_text(args.output_json, rendered + "\n")
        print(f"Wrote timestamp JSON: {output_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

