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
"""Audio placeholders in OOMptimizer's synthetic batches.

A model that interleaves audio into the LLM sequence finds it by scanning the input ids for a
negative sentinel. ``torch.randint(0, vocab_size)`` can never produce one, so before this the
synthetic batch was pure text: the encoder output was discarded, the encoder stayed out of the
backward graph, and the profile omitted its activations, gradients and optimizer state.

Two properties are load-bearing and tested here:

* the generator plants the sentinel, AND the resolver lengthens the sequence to hold it (planting
  alone would just relabel text positions, leaving the sequence about half its real length);
* schemas that do not declare the keys are byte-for-byte unaffected, because SALM, DuplexS2S and
  the ASR models share this code path.
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from nemo.collections.speechlm2.models import streaming_stt_model as stt
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils.oomptimizer import (
    SequenceLengthResolver,
    audio_placeholder_frames,
    fill_audio_placeholders,
    schema_audio_frame_stride,
)

SCRIPTS = Path(__file__).parents[3] / "scripts" / "speechlm2"
PLACEHOLDER_ID = -200
IGNORE_ID = -100
STRIDE = 1280  # 0.08 s frames at 16 kHz
TEN_SECONDS = 160000


@dataclass
class _Batch:
    input_tokens: torch.Tensor
    audios: torch.Tensor
    audio_lens: torch.Tensor
    chunk_size: int | None = None


def _schema(with_audio_keys: bool = True, cls=_Batch) -> dict:
    labels = {
        "name": "input_tokens",
        "type": NeuralType(("B", "T"), LabelsType()),
        "seq_length": "output",
        "vocab_size": 100,
    }
    if with_audio_keys:
        labels["audio_placeholder_id"] = PLACEHOLDER_ID
        labels["audio_frame_stride_samples"] = STRIDE
    return {
        "cls": cls,
        "inputs": [
            labels,
            {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
            {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
        ],
    }


def _resolver(schema) -> SequenceLengthResolver:
    return SequenceLengthResolver(cfg=None, ratio=12, salm_audio_token_ratio=0.75, schema=schema)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["oomptimizer", "distributed_oomptimizer"])
def generator_cls(request):
    """Both scripts carry a copy of ProfilingBatchGenerator; neither may drift from the other."""
    return _load_script(request.param).ProfilingBatchGenerator


# ===========================================================================
# Frame arithmetic
# ===========================================================================


def test_frames_for_a_ten_second_bucket():
    assert audio_placeholder_frames(STRIDE, TEN_SECONDS) == 125


def test_frames_is_zero_without_a_declared_stride():
    assert audio_placeholder_frames(None, TEN_SECONDS) == 0
    assert audio_placeholder_frames(0, TEN_SECONDS) == 0


def test_frames_is_zero_for_empty_audio():
    assert audio_placeholder_frames(STRIDE, 0) == 0


def test_sub_frame_audio_still_occupies_one_position():
    """Floor would give 0 and silently restore the pure-text batch this guards against."""
    assert audio_placeholder_frames(STRIDE, STRIDE - 1) == 1


def test_stride_is_read_from_the_schema():
    assert schema_audio_frame_stride(_schema()) == STRIDE
    assert schema_audio_frame_stride(_schema(with_audio_keys=False)) == 0
    assert schema_audio_frame_stride(None) == 0


# ===========================================================================
# Filling
# ===========================================================================


def test_fill_writes_a_contiguous_leading_run():
    tensor = torch.randint(0, 100, (2, 245))
    written = fill_audio_placeholders(tensor, _schema()["inputs"][0], TEN_SECONDS)
    assert written == 125
    assert (tensor[:, :125] == PLACEHOLDER_ID).all()
    assert (tensor[:, 125:] != PLACEHOLDER_ID).all()


def test_fill_is_a_noop_without_the_keys():
    tensor = torch.randint(0, 100, (2, 245))
    before = tensor.clone()
    assert fill_audio_placeholders(tensor, _schema(with_audio_keys=False)["inputs"][0], TEN_SECONDS) == 0
    assert torch.equal(tensor, before)


def test_fill_clamps_to_the_tensor_width():
    """A bucket whose audio outruns the resolved sequence must not raise or wrap."""
    tensor = torch.randint(0, 100, (2, 10))
    assert fill_audio_placeholders(tensor, _schema()["inputs"][0], TEN_SECONDS) == 10
    assert (tensor == PLACEHOLDER_ID).all()


# ===========================================================================
# Resolver
# ===========================================================================


def test_resolver_makes_room_for_the_audio_positions():
    audio_samples, output_len = _resolver(_schema()).resolve_one(10.0)
    assert audio_samples == TEN_SECONDS
    # 120 text tokens (ratio 12 x 10 s) + 125 audio positions.
    assert output_len == 245


def test_resolver_is_unchanged_for_schemas_without_the_keys():
    """Regression guard for SALM / DuplexS2S / ASR, which share this resolver."""
    assert _resolver(_schema(with_audio_keys=False)).resolve_one(10.0) == (TEN_SECONDS, 120)


def test_resolver_and_generator_agree_on_the_frame_count():
    """If these two disagree the placeholders overflow or under-fill the sequence."""
    schema = _schema()
    audio_samples, output_len = _resolver(schema).resolve_one(10.0)
    tensor = torch.randint(0, 100, (2, output_len))
    written = fill_audio_placeholders(tensor, schema["inputs"][0], audio_samples)
    assert 0 < written < output_len


# ===========================================================================
# Generators in both scripts
# ===========================================================================


def test_generator_plants_the_sentinel(generator_cls):
    schema = _schema()
    gen = generator_cls(schema=schema, start_batch_size=2, device="cpu")
    batch = gen(*_resolver(schema).resolve_one(10.0))
    assert (batch.input_tokens == PLACEHOLDER_ID).any()
    assert (batch.input_tokens == PLACEHOLDER_ID).sum(dim=1).tolist() == [125, 125]


def test_generator_leaves_other_schemas_pure_text(generator_cls):
    schema = _schema(with_audio_keys=False)
    gen = generator_cls(schema=schema, start_batch_size=2, device="cpu")
    batch = gen(*_resolver(schema).resolve_one(10.0))
    assert not (batch.input_tokens < 0).any()


def test_generator_passes_scalars_through_unwrapped(generator_cls):
    """``constant`` wraps its value in a 1-element tensor; fields like chunk_size need an int."""
    schema = _schema()
    schema["inputs"].append({"name": "chunk_size", "type": "scalar", "value": 28})
    gen = generator_cls(schema=schema, start_batch_size=2, device="cpu")
    batch = gen(*_resolver(schema).resolve_one(10.0))
    assert batch.chunk_size == 28
    assert not torch.is_tensor(batch.chunk_size)


# ===========================================================================
# The model-side guard
# ===========================================================================


def test_dropping_encoder_output_is_logged(monkeypatch, caplog):
    """A sequence with no audio positions silently discards the encoder; say so once."""
    monkeypatch.setattr(stt, "_WARNED_AUDIO_DROPPED", False)
    input_tokens = torch.full((2, 6), 5, dtype=torch.long)
    with caplog.at_level("WARNING"):
        out = stt.interleave_embeddings(
            input_tokens=input_tokens,
            audio_mask=torch.zeros_like(input_tokens, dtype=torch.bool),
            text_embeds=torch.zeros(2, 6, 4),
            audio_embs=torch.zeros(2, 3, 4),
            pad_id=0,
        )
    assert "no AUDIO_TOKEN_IDX positions" in caplog.text
    assert out["input_embeds"].shape == (2, 6, 4)


def test_genuine_pure_text_is_not_logged(monkeypatch, caplog):
    """No encoder frames means no audio was dropped -- do not cry wolf."""
    monkeypatch.setattr(stt, "_WARNED_AUDIO_DROPPED", False)
    input_tokens = torch.full((2, 6), 5, dtype=torch.long)
    with caplog.at_level("WARNING"):
        stt.interleave_embeddings(
            input_tokens=input_tokens,
            audio_mask=torch.zeros_like(input_tokens, dtype=torch.bool),
            text_embeds=torch.zeros(2, 6, 4),
            audio_embs=torch.zeros(2, 0, 4),
            pad_id=0,
        )
    assert "no AUDIO_TOKEN_IDX positions" not in caplog.text
