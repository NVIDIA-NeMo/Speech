# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import json
import math
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import click
import torch
from omegaconf import DictConfig, ListConfig

_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".pcm",
    ".sph",
    ".wav",
    ".wma",
}


def configured_audio_path_resolution_modes(config) -> tuple[str, ...]:
    """Return stable resolution-mode labels derived only from the resolved config."""
    modes: set[str] = set()

    def visit(node) -> None:
        if isinstance(node, (list, tuple, ListConfig)):
            for item in node:
                visit(item)
            return
        if not isinstance(node, (dict, DictConfig)):
            return

        typ = node.get("type")
        if typ == "share_gpt_webdataset":
            if int(node.get("wds_sample_index_version", 1)) == 2:
                modes.add("wds_member_exact_then_unambiguous_basename")
            else:
                modes.add("wds_legacy_single_audio")
        elif typ == "share_gpt":
            lookup_mode = node.get("tar_lookup_mode")
            if lookup_mode == "collection":
                modes.add("tar_collection_route")
            elif lookup_mode == "paired" or node.get("tarred_audio_filepaths") is not None:
                modes.add("paired_manifest_tar")
            else:
                modes.add("direct_or_url")
            if node.get("audio_path_prefix_map"):
                modes.add("prefix_map")
        elif typ in {"multimodal_conversation", "nemo", "nemo_tarred"}:
            modes.add("manifest_audio_reference")

        for value in node.values():
            visit(value)

    visit(config)
    return tuple(sorted(modes))


def _codec_from_cut(cut) -> str | None:
    custom = getattr(cut, "custom", None) or {}
    custom_codec = custom.get("_source_codec")
    if isinstance(custom_codec, str) and custom_codec:
        return custom_codec
    recording = getattr(cut, "recording", None)
    sources = getattr(recording, "sources", ()) if recording is not None else ()
    for source in sources:
        value = getattr(source, "source", None)
        if not isinstance(value, (str, Path)):
            continue
        path = urlparse(str(value)).path
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in _AUDIO_SUFFIXES:
            return suffix.removeprefix(".")
    return None


def _source_range_from_cut(cut) -> tuple[str, int] | None:
    custom = getattr(cut, "custom", None) or {}
    key = custom.get("_source_read_key")
    num_bytes = _scalar_int(custom.get("_source_range_bytes"))
    if not isinstance(key, str) or not key or num_bytes is None:
        return None
    return key, num_bytes


def _scalar_int(value) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def _nearest_rank_percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


@dataclass
class FullValidationStats:
    """Aggregate content-free measurements from successfully materialized SALM batches."""

    requested_batches: int | None
    audio_placeholder_token_id: int | None = None
    audio_path_resolution_modes: tuple[str, ...] = ()
    completed_batches: int = 0
    examples: int = 0
    conversations: int = 0
    audio_items: int = 0
    decoded_seconds: float = 0.0
    zero_audio_samples: int = 0
    one_audio_samples: int = 0
    multi_audio_samples: int = 0
    audio_turn_placeholders: int = 0
    tokenized_audio_placeholders: int = 0
    sample_rates: set[int] = field(default_factory=set)
    channel_counts: set[int] = field(default_factory=set)
    codecs: set[str] = field(default_factory=set)
    unknown_codec_items: int = 0
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    first_batch_latency_ms: float | None = None
    steady_batch_latencies_ms: list[float] = field(default_factory=list)
    measured_bytes_read: int = 0
    measured_bytes_batches: int = 0
    unavailable_bytes_batches: int = 0
    failures: list[dict] = field(default_factory=list)

    def observe_batch(self, batch: dict, *, latency_ms: float) -> None:
        from nemo.collections.common.data.lhotse.text_adapters import AudioTurn

        conversations = list(batch["conversations"])
        audio_cuts = []
        audio_counts = []
        for conversation in conversations:
            cuts = [turn.cut for turn in conversation.turns if isinstance(turn, AudioTurn)]
            audio_counts.append(len(cuts))
            audio_cuts.extend(cuts)

        audio_lens_value = batch["audio_lens"]
        if torch.is_tensor(audio_lens_value):
            audio_lens = [int(value) for value in audio_lens_value.detach().cpu().flatten().tolist()]
        else:
            audio_lens = [int(value) for value in audio_lens_value]
        if len(audio_lens) != len(audio_cuts):
            raise click.ClickException(
                "full validation audio_lens/audio-turn mismatch: "
                f"audio_lens={len(audio_lens)} audio_turns={len(audio_cuts)}"
            )

        placeholder_count = len(audio_cuts)
        if self.audio_placeholder_token_id is not None:
            input_ids = batch["input_ids"]
            if not torch.is_tensor(input_ids):
                input_ids = torch.as_tensor(input_ids)
            tokenized_count = int((input_ids == self.audio_placeholder_token_id).sum().item())
            if tokenized_count != placeholder_count:
                raise click.ClickException(
                    "full validation placeholder/audio-turn mismatch: "
                    f"tokenized_placeholders={tokenized_count} audio_turns={placeholder_count}"
                )
            self.tokenized_audio_placeholders += tokenized_count

        decoded_durations = []
        source_ranges: dict[str, int] = {}
        missing_source_range = False
        for cut, num_samples in zip(audio_cuts, audio_lens):
            sample_rate = int(cut.sampling_rate)
            channels = int(cut.num_channels)
            if sample_rate <= 0 or channels <= 0 or num_samples < 0:
                raise click.ClickException("full validation observed invalid decoded audio metadata")
            duration = num_samples / sample_rate
            decoded_durations.append(duration)
            self.sample_rates.add(sample_rate)
            self.channel_counts.add(channels)
            if (codec := _codec_from_cut(cut)) is None:
                self.unknown_codec_items += 1
            else:
                self.codecs.add(codec)
            source_range = _source_range_from_cut(cut)
            if source_range is None:
                missing_source_range = True
            else:
                key, num_bytes = source_range
                previous = source_ranges.setdefault(key, num_bytes)
                if previous != num_bytes:
                    raise click.ClickException(
                        "full validation observed inconsistent byte counts for one source range"
                    )

        self.examples += len(conversations)
        self.conversations += len(conversations)
        self.audio_items += len(audio_cuts)
        self.decoded_seconds += sum(decoded_durations)
        self.zero_audio_samples += sum(count == 0 for count in audio_counts)
        self.one_audio_samples += sum(count == 1 for count in audio_counts)
        self.multi_audio_samples += sum(count > 1 for count in audio_counts)
        self.audio_turn_placeholders += placeholder_count
        if decoded_durations:
            batch_min = min(decoded_durations)
            batch_max = max(decoded_durations)
            self.min_duration_seconds = (
                batch_min if self.min_duration_seconds is None else min(self.min_duration_seconds, batch_min)
            )
            self.max_duration_seconds = (
                batch_max if self.max_duration_seconds is None else max(self.max_duration_seconds, batch_max)
            )

        measured_bytes = _scalar_int(batch.get("bytes_read"))
        if measured_bytes is not None:
            self.measured_bytes_read += measured_bytes
            self.measured_bytes_batches += 1
        elif source_ranges:
            self.measured_bytes_read += sum(source_ranges.values())
            self.measured_bytes_batches += 1
            if missing_source_range:
                self.unavailable_bytes_batches += 1
        elif audio_cuts:
            self.unavailable_bytes_batches += 1
        else:
            self.measured_bytes_batches += 1

        latency_ms = float(latency_ms)
        if self.completed_batches == 0:
            self.first_batch_latency_ms = latency_ms
        else:
            self.steady_batch_latencies_ms.append(latency_ms)
        self.completed_batches += 1

    def record_failure(self, *, step: int, error: Exception) -> None:
        self.failures.append(
            {
                "step": int(step),
                "stage": "materialize_or_measure",
                "error_type": type(error).__name__,
            }
        )

    def _codec_summary(self) -> dict:
        if self.audio_items == 0:
            status = "not_applicable"
        elif self.unknown_codec_items == self.audio_items:
            status = "unavailable"
        elif self.unknown_codec_items:
            status = "partial"
        else:
            status = "measured"
        return {
            "status": status,
            "values": sorted(self.codecs),
            "unknown_items": self.unknown_codec_items,
        }

    def _bytes_summary(self) -> dict:
        if self.measured_bytes_batches == 0:
            return {
                "status": "unavailable",
                "value": None,
                "reason": "source byte counters are not exposed by the materialized batch",
            }
        if self.unavailable_bytes_batches:
            return {
                "status": "partial",
                "value": self.measured_bytes_read,
                "unavailable_batches": self.unavailable_bytes_batches,
            }
        return {"status": "measured", "value": self.measured_bytes_read}

    def to_summary(self, *, phase: str, rank: int, world_size: int, status: str) -> dict:
        steady = self.steady_batch_latencies_ms
        placeholder_status = "measured" if self.audio_placeholder_token_id is not None else "unavailable"
        resolution_modes = sorted(set(self.audio_path_resolution_modes))
        return {
            "schema_version": 1,
            "mode": "full",
            "phase": phase,
            "rank": int(rank),
            "world_size": int(world_size),
            "status": status,
            "requested_batches": self.requested_batches,
            "completed_batches": self.completed_batches,
            "counters": {
                "examples": self.examples,
                "conversations": self.conversations,
                "audio_items": self.audio_items,
                "decoded_seconds": round(self.decoded_seconds, 6),
                "zero_audio_samples": self.zero_audio_samples,
                "one_audio_samples": self.one_audio_samples,
                "multi_audio_samples": self.multi_audio_samples,
            },
            "audio": {
                "codecs": self._codec_summary(),
                "sample_rates_hz": sorted(self.sample_rates),
                "channel_counts": sorted(self.channel_counts),
                "duration_seconds": {
                    "min": round(self.min_duration_seconds, 6) if self.min_duration_seconds is not None else None,
                    "max": round(self.max_duration_seconds, 6) if self.max_duration_seconds is not None else None,
                },
                "placeholder_counts": {
                    "audio_turns": self.audio_turn_placeholders,
                    "tokenized_audio_placeholders": (
                        self.tokenized_audio_placeholders if self.audio_placeholder_token_id is not None else None
                    ),
                    "status": placeholder_status,
                },
                "path_resolution_modes": {
                    "status": "configured" if resolution_modes else "unavailable",
                    "values": resolution_modes,
                },
            },
            "latency_ms": {
                "first_batch": (
                    round(self.first_batch_latency_ms, 3) if self.first_batch_latency_ms is not None else None
                ),
                "p50_batch": round(statistics.median(steady), 3) if steady else None,
                "p95_batch": (round(_nearest_rank_percentile(steady, 0.95), 3) if steady else None),
                "steady_batch_count": len(steady),
            },
            "bytes_read": self._bytes_summary(),
            "failures": list(self.failures),
        }

    def write(self, path, *, phase: str, rank: int, world_size: int, status: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.to_summary(phase=phase, rank=rank, world_size=world_size, status=status)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


@contextmanager
def full_summary_failure_guard(
    stats: FullValidationStats | None,
    path,
    *,
    phase: str,
    rank: int,
    world_size: int,
):
    """Write a content-free failed summary before re-raising iteration errors."""
    try:
        yield
    except Exception as error:
        if stats is not None and path is not None:
            stats.record_failure(step=stats.completed_batches, error=error)
            stats.write(
                path,
                phase=phase,
                rank=rank,
                world_size=world_size,
                status="failed",
            )
        raise
