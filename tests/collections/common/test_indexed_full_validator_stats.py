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

"""Coverage for content-free full-validation statistics."""

import json

import pytest
import torch
from lhotse import AudioSource, Recording
from lhotse.utils import fastcopy
from omegaconf import OmegaConf
from scripts.dataloading._validate_dataloader.full_stats import (
    FullValidationStats,
    configured_audio_path_resolution_modes,
)

from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, NeMoMultimodalConversation, TextTurn


def _cut(recording_id, source, *, sampling_rate, num_samples, channels):
    recording = Recording(
        id=recording_id,
        sources=[AudioSource(type="file", channels=list(range(channels)), source=source)],
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        duration=num_samples / sampling_rate,
    )
    return recording.to_cut()


def _batch():
    one = AudioTurn(
        cut=_cut("one", "/audio/one.wav", sampling_rate=16000, num_samples=16000, channels=1),
        role="user",
        audio_locator_tag="<audio>",
    )
    multi_a = AudioTurn(
        cut=_cut("multi-a", "/audio/a.flac", sampling_rate=8000, num_samples=4000, channels=2),
        role="user",
        audio_locator_tag="<audio>",
    )
    multi_b = AudioTurn(
        cut=_cut("multi-b", b"encoded", sampling_rate=16000, num_samples=4000, channels=1),
        role="user",
        audio_locator_tag="<audio>",
    )
    conversations = [
        NeMoMultimodalConversation("zero", [TextTurn("hello", "user")]),
        NeMoMultimodalConversation("one", [one, TextTurn("answer", "assistant")]),
        NeMoMultimodalConversation("multi", [multi_a, multi_b, TextTurn("answer", "assistant")]),
    ]
    return {
        "audios": torch.zeros(3, 16000),
        "audio_lens": torch.tensor([16000, 4000, 4000]),
        "input_ids": torch.tensor([[0, 0, 1], [0, 7, 1], [7, 7, 1]]),
        "loss_mask": torch.ones(3, 3, dtype=torch.bool),
        "conversations": conversations,
    }


def test_full_stats_aggregates_materialized_audio_without_contents(tmp_path):
    stats = FullValidationStats(
        requested_batches=3,
        audio_placeholder_token_id=7,
        audio_path_resolution_modes=("tar_collection_route",),
    )
    stats.observe_batch(_batch(), latency_ms=100.0)
    stats.observe_batch(_batch(), latency_ms=10.0)
    stats.observe_batch(_batch(), latency_ms=30.0)
    output = tmp_path / "summary.json"
    stats.write(output, phase="baseline", rank=0, world_size=1, status="passed")

    summary = json.loads(output.read_text())
    assert summary["requested_batches"] == 3
    assert summary["completed_batches"] == 3
    assert summary["counters"] == {
        "examples": 9,
        "conversations": 9,
        "audio_items": 9,
        "decoded_seconds": 5.25,
        "zero_audio_samples": 3,
        "one_audio_samples": 3,
        "multi_audio_samples": 3,
    }
    assert summary["audio"]["sample_rates_hz"] == [8000, 16000]
    assert summary["audio"]["channel_counts"] == [1, 2]
    assert summary["audio"]["duration_seconds"] == {"min": 0.25, "max": 1.0}
    assert summary["audio"]["codecs"] == {
        "status": "partial",
        "values": ["flac", "wav"],
        "unknown_items": 3,
    }
    assert summary["audio"]["placeholder_counts"] == {
        "audio_turns": 9,
        "tokenized_audio_placeholders": 9,
        "status": "measured",
    }
    assert summary["audio"]["path_resolution_modes"] == {
        "status": "configured",
        "values": ["tar_collection_route"],
    }
    assert summary["latency_ms"] == {
        "first_batch": 100.0,
        "p50_batch": 20.0,
        "p95_batch": 30.0,
        "steady_batch_count": 2,
    }
    assert summary["bytes_read"]["status"] == "unavailable"
    assert summary["failures"] == []
    assert "hello" not in output.read_text()
    assert "answer" not in output.read_text()


def test_full_stats_rejects_placeholder_or_audio_length_mismatch():
    batch = _batch()
    batch["input_ids"] = torch.zeros_like(batch["input_ids"])
    stats = FullValidationStats(requested_batches=1, audio_placeholder_token_id=7)
    with pytest.raises(Exception, match="placeholder"):
        stats.observe_batch(batch, latency_ms=1.0)

    batch = _batch()
    batch["audio_lens"] = torch.tensor([16000])
    with pytest.raises(Exception, match="audio_lens"):
        stats.observe_batch(batch, latency_ms=1.0)


def test_full_stats_records_content_free_failure_and_measured_bytes(tmp_path):
    batch = _batch()
    batch["bytes_read"] = 123
    stats = FullValidationStats(requested_batches=2, audio_placeholder_token_id=7)
    stats.observe_batch(batch, latency_ms=5.0)
    stats.record_failure(step=1, error=RuntimeError("secret conversation contents"))
    output = tmp_path / "failed.json"
    stats.write(output, phase="baseline", rank=2, world_size=4, status="failed")

    summary = json.loads(output.read_text())
    assert summary["bytes_read"] == {"status": "measured", "value": 123}
    assert summary["failures"] == [{"step": 1, "stage": "materialize_or_measure", "error_type": "RuntimeError"}]
    assert "secret conversation contents" not in output.read_text()


def test_resolution_modes_are_derived_from_config_without_scanning_data():
    cfg = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "share_gpt_webdataset",
                    "wds_sample_index_version": 2,
                },
                {
                    "type": "share_gpt",
                    "tar_lookup_mode": "collection",
                    "audio_path_prefix_map": {"/old": "/new"},
                },
                {"type": "share_gpt"},
            ]
        }
    )

    assert configured_audio_path_resolution_modes(cfg) == (
        "direct_or_url",
        "prefix_map",
        "tar_collection_route",
        "wds_member_exact_then_unambiguous_basename",
    )


def test_full_stats_measure_memory_codec_and_deduplicated_source_ranges():
    repeated = _cut("repeated", b"encoded", sampling_rate=16000, num_samples=16000, channels=1)
    repeated = fastcopy(
        repeated,
        custom={
            "_source_codec": "wav",
            "_source_range_bytes": 2048,
            "_source_read_key": "s3://bucket/sample.tar@0:2048",
        },
    )
    other = _cut("other", b"encoded", sampling_rate=16000, num_samples=8000, channels=1)
    other = fastcopy(
        other,
        custom={
            "_source_codec": "flac",
            "_source_range_bytes": 1024,
            "_source_read_key": "s3://bucket/sample.tar@2048:3072",
        },
    )
    conversation = NeMoMultimodalConversation(
        "sample",
        [
            AudioTurn(repeated, "user", "<audio>"),
            AudioTurn(repeated, "user", "<audio>"),
            AudioTurn(other, "user", "<audio>"),
        ],
    )
    batch = {
        "audios": torch.zeros(3, 16000),
        "audio_lens": torch.tensor([16000, 16000, 8000]),
        "input_ids": torch.tensor([[7, 7, 7]]),
        "loss_mask": torch.ones(1, 3, dtype=torch.bool),
        "conversations": [conversation],
    }
    stats = FullValidationStats(requested_batches=1, audio_placeholder_token_id=7)

    stats.observe_batch(batch, latency_ms=1.0)
    summary = stats.to_summary(phase="baseline", rank=0, world_size=1, status="passed")

    assert summary["audio"]["codecs"] == {
        "status": "measured",
        "values": ["flac", "wav"],
        "unknown_items": 0,
    }
    assert summary["bytes_read"] == {"status": "measured", "value": 3072}
