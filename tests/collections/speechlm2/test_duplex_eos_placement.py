# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Tests for agent EOS placement logic in DuplexSTTDataset.

When source tokens are collated with agent_token_channel provided, the agent EOS
should be placed in the agent (target) token channel at the correct position:
user_turn_start + eos_offset_frames after each user turn.
"""

import os

import pytest
import torch
from lhotse import CutSet, SupervisionSegment, compute_num_frames
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.duplex_stt_dataset import (
    DuplexSTTDataset,
    collate_token_channel,
)
from nemo.collections.speechlm2.data.utils import get_pad_id

SR = 16000
FL = 0.08


@pytest.fixture(scope="session")
def tokenizer():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        model_path = "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1"
    else:
        model_path = "TinyLlama/TinyLlama_v1.1"
    return AutoTokenizer(model_path, use_fast=True)


@pytest.fixture(scope="session")
def single_turn_cuts():
    """Single cut with one user turn and one agent turn."""
    cut = dummy_cut(0, duration=2.0, recording=dummy_recording(0, duration=2.0, with_data=True))
    cut.supervisions = [
        SupervisionSegment(
            id="user-0", recording_id=cut.recording_id, start=0.0, duration=0.5, text="hello", speaker="user"
        ),
        SupervisionSegment(
            id="agent-0", recording_id=cut.recording_id, start=0.6, duration=0.5, text="hi there", speaker="assistant"
        ),
    ]
    return CutSet([cut])


@pytest.fixture(scope="session")
def multi_turn_cuts():
    """Single cut with two user turns and two agent turns."""
    cut = dummy_cut(0, duration=4.0, recording=dummy_recording(0, duration=4.0, with_data=True))
    cut.supervisions = [
        SupervisionSegment(
            id="user-0", recording_id=cut.recording_id, start=0.0, duration=0.5, text="hello", speaker="user"
        ),
        SupervisionSegment(
            id="agent-0", recording_id=cut.recording_id, start=0.6, duration=0.5, text="hi", speaker="assistant"
        ),
        SupervisionSegment(
            id="user-1", recording_id=cut.recording_id, start=1.5, duration=0.5, text="how are you", speaker="user"
        ),
        SupervisionSegment(
            id="agent-1", recording_id=cut.recording_id, start=2.2, duration=0.5, text="good thanks", speaker="assistant"
        ),
    ]
    return CutSet([cut])


def test_agent_eos_placed_in_target_channel(tokenizer, single_turn_cuts):
    """Agent EOS should be placed in target_tokens when source collate triggers it."""
    bos = tokenizer.bos
    eos = tokenizer.eos
    pad = get_pad_id(tokenizer)

    # First, collate target tokens WITHOUT EOS (skip_eos=True)
    target_tokens, target_token_lens = collate_token_channel(
        single_turn_cuts,
        tokenizer,
        frame_length=FL,
        roles={"assistant"},
        bos_id=bos,
        eos_id=eos,
        pad_id=pad,
        remove_timestamps=True,
        skip_eos=True,
    )

    # Verify no EOS in target tokens yet
    assert (target_tokens == eos).sum().item() == 0, "skip_eos=True should not place any EOS"

    # Now collate source tokens, passing in the target channel for EOS placement
    source_tokens, source_token_lens = collate_token_channel(
        single_turn_cuts,
        tokenizer,
        frame_length=FL,
        roles={"user"},
        bos_id=bos,
        eos_id=eos,
        pad_id=pad,
        remove_timestamps=True,
        agent_token_channel=target_tokens,
        agent_token_channel_lengths=target_token_lens.clone(),
        agent_eos_id=eos,
    )

    # Agent EOS should now be placed in target_tokens
    eos_positions = (target_tokens[0] == eos).nonzero(as_tuple=True)[0]
    assert len(eos_positions) > 0, "Agent EOS should be placed in target_tokens after source collation"

    # EOS should be at user_start + eos_offset_frames (default 8)
    user_start_frame = compute_num_frames(0.0, FL, SR)  # = 0
    expected_eos_pos = user_start_frame + 8
    assert expected_eos_pos in eos_positions.tolist(), (
        f"Agent EOS expected at frame {expected_eos_pos}, found at {eos_positions.tolist()}"
    )


def test_agent_eos_placed_for_each_user_turn(tokenizer, multi_turn_cuts):
    """Each user turn should trigger an agent EOS placement in the target channel."""
    bos = tokenizer.bos
    eos = tokenizer.eos
    pad = get_pad_id(tokenizer)

    target_tokens, target_token_lens = collate_token_channel(
        multi_turn_cuts,
        tokenizer,
        frame_length=FL,
        roles={"assistant"},
        bos_id=bos,
        eos_id=eos,
        pad_id=pad,
        remove_timestamps=True,
        skip_eos=True,
    )

    source_tokens, source_token_lens = collate_token_channel(
        multi_turn_cuts,
        tokenizer,
        frame_length=FL,
        roles={"user"},
        bos_id=bos,
        eos_id=eos,
        pad_id=pad,
        remove_timestamps=True,
        agent_token_channel=target_tokens,
        agent_token_channel_lengths=target_token_lens.clone(),
        agent_eos_id=eos,
    )

    eos_positions = (target_tokens[0] == eos).nonzero(as_tuple=True)[0].tolist()

    # Two user turns → expect two agent EOS placements
    user_turn_starts = [0.0, 1.5]
    expected_eos_positions = [compute_num_frames(s, FL, SR) + 8 for s in user_turn_starts]
    # Filter out positions that are out of bounds
    expected_eos_positions = [p for p in expected_eos_positions if p < target_token_lens[0].item()]

    for expected_pos in expected_eos_positions:
        assert expected_pos in eos_positions, (
            f"Agent EOS expected at frame {expected_pos} but not found. "
            f"Actual EOS positions: {eos_positions}"
        )


def test_skip_eos_true_produces_no_eos_in_target(tokenizer, single_turn_cuts):
    """When skip_eos=True and no agent_token_channel, target should have no EOS."""
    bos = tokenizer.bos
    eos = tokenizer.eos
    pad = get_pad_id(tokenizer)

    target_tokens, _ = collate_token_channel(
        single_turn_cuts,
        tokenizer,
        frame_length=FL,
        roles={"assistant"},
        bos_id=bos,
        eos_id=eos,
        pad_id=pad,
        remove_timestamps=True,
        skip_eos=True,
    )

    assert (target_tokens == eos).sum().item() == 0, "skip_eos=True should not place EOS"


def test_skip_eos_false_places_eos_at_supervision_end(tokenizer, single_turn_cuts):
    """When skip_eos=False (default), EOS should be at the supervision end frame."""
    bos = tokenizer.bos
    eos = tokenizer.eos
    pad = get_pad_id(tokenizer)

    target_tokens, _ = collate_token_channel(
        single_turn_cuts,
        tokenizer,
        frame_length=FL,
        roles={"assistant"},
        bos_id=bos,
        eos_id=eos,
        pad_id=pad,
        remove_timestamps=True,
        skip_eos=False,
    )

    # Agent turn: start=0.6, duration=0.5 → end=1.1
    eos_frame = compute_num_frames(1.1, FL, SR)
    total_frames = compute_num_frames(2.0, FL, SR)
    if eos_frame < total_frames:
        assert target_tokens[0, eos_frame].item() == eos, (
            f"EOS expected at frame {eos_frame}, got token {target_tokens[0, eos_frame].item()}"
        )


def test_end_to_end_eos_in_dataset(tokenizer, single_turn_cuts):
    """End-to-end test: DuplexSTTDataset should place agent EOS in target_tokens."""
    dataset = DuplexSTTDataset(
        tokenizer=tokenizer,
        frame_length=FL,
        source_sample_rate=SR,
        input_roles=["user"],
        output_roles=["assistant"],
        cfg={"prepend_word_space": False},
        model_cfg={"predict_user_text": False},
        is_training=True,
    )

    batch = dataset[single_turn_cuts]
    target_tokens = batch["audio_data"]["target_tokens"]
    eos = tokenizer.eos

    eos_positions = (target_tokens[0] == eos).nonzero(as_tuple=True)[0].tolist()
    assert len(eos_positions) > 0, (
        "Agent EOS should appear in target_tokens from end-to-end dataset output"
    )
