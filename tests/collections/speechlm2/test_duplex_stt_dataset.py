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

import os
import re

import pytest
import torch
from lhotse import CutSet, SupervisionSegment, compute_num_frames
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.duplex_stt_dataset import (
    DuplexSTTDataset,
    collate_system_prompt,
    collate_token_channel,
)
from nemo.collections.speechlm2.data.utils import get_pad_id

SR = 16000
FL = 0.08


def _clean_text(text):
    """Strip timestamp tokens and normalize whitespace, matching _text_to_ids(remove_timestamps=True)."""
    text = re.sub(r'<\|\d+\|>', '', text)
    return ' '.join(text.strip().split())


def _verify_supervision_tokens(tokens_1d, start, duration, raw_text, tokenizer, pad, bos, eos, total_frames):
    """Verify BOS/EOS placement and that decoded token IDs match the original supervision text.

    Checks:
        1. BOS is placed at the frame corresponding to supervision start.
        2. EOS is placed at the frame corresponding to supervision end (if within cut).
        3. Text token IDs between BOS and EOS are a prefix of tokenizer.text_to_ids(clean_text).
        4. Decoding those IDs back to text matches (or is a prefix of) the original text.
    """
    pos = compute_num_frames(start, FL, SR)
    eospos = compute_num_frames(start + duration, FL, SR)
    clean = _clean_text(raw_text)

    # 1. BOS at turn start
    assert tokens_1d[pos].item() == bos, f"Expected BOS={bos} at frame {pos}, got {tokens_1d[pos].item()}"

    # 2. EOS at turn end (only if within cut bounds)
    if eospos < total_frames:
        assert tokens_1d[eospos].item() == eos, f"Expected EOS={eos} at frame {eospos}, got {tokens_1d[eospos].item()}"

    # 3. Extract text token IDs between BOS and EOS, filtering out pad
    end = min(eospos, total_frames)
    actual_ids = [t for t in tokens_1d[pos + 1 : end].tolist() if t != pad]
    expected_ids = tokenizer.text_to_ids(clean)

    # Actual IDs should be a prefix of expected (truncation may occur when text is long)
    assert actual_ids == expected_ids[: len(actual_ids)], (
        f"Token ID mismatch for '{clean}':\n"
        f"  actual_ids      = {actual_ids}\n"
        f"  expected_prefix = {expected_ids[: len(actual_ids)]}\n"
        f"  full_expected   = {expected_ids}"
    )

    # 4. Decode IDs back to text and verify against original
    if actual_ids:
        decoded = tokenizer.ids_to_text(actual_ids).strip()
        if len(actual_ids) == len(expected_ids):
            assert decoded == clean, f"Full decode mismatch: '{decoded}' != '{clean}', ids={actual_ids}"
        else:
            assert clean.startswith(decoded), (
                f"Truncated decode '{decoded}' is not a prefix of '{clean}'\n"
                f"  ids={actual_ids} (truncated {len(expected_ids)} → {len(actual_ids)})"
            )


@pytest.fixture(scope="session")
def tokenizer():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        model_path = "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1"
    else:
        model_path = "TinyLlama/TinyLlama_v1.1"
    return AutoTokenizer(model_path, use_fast=True)


@pytest.fixture(scope="session")
def cuts():
    """Two cuts: cut1 with plain text, cut2 with timestamped text and a system prompt."""
    cut1 = dummy_cut(0, duration=1.0, recording=dummy_recording(0, duration=1.0, with_data=True))
    cut1.supervisions = [
        SupervisionSegment(
            id="s0-user", recording_id=cut1.recording_id, start=0, duration=0.3, text="hi", speaker="user"
        ),
        SupervisionSegment(
            id="s0-agent", recording_id=cut1.recording_id, start=0.4, duration=0.3, text="hello", speaker="assistant"
        ),
    ]

    cut2 = dummy_cut(1, duration=2.0, recording=dummy_recording(1, duration=2.0, with_data=True))
    cut2.supervisions = [
        SupervisionSegment(
            id="s1-user",
            recording_id=cut2.recording_id,
            start=0,
            duration=0.5,
            text="<|0|> good <|1|> <|3|> morning <|5|>",
            speaker="user",
        ),
        SupervisionSegment(
            id="s1-agent",
            recording_id=cut2.recording_id,
            start=0.6,
            duration=0.5,
            text="<|0|> good <|2|> <|2|> morning <|4|> <|4|> to <|5|> <|5|> you <|6|>",
            speaker="assistant",
        ),
        SupervisionSegment(
            id="s1-user2",
            recording_id=cut2.recording_id,
            start=1.2,
            duration=0.3,
            text="<|0|> thanks <|3|>",
            speaker="user",
        ),
        SupervisionSegment(
            id="s1-agent2",
            recording_id=cut2.recording_id,
            start=1.6,
            duration=0.4,
            text="<|0|> welcome <|4|>",
            speaker="assistant",
        ),
    ]
    cut2.custom = {"system_prompt": "be helpful"}

    return CutSet([cut1, cut2])


def test_collate_audio(cuts):
    """Test collate_audio: shapes, lengths, and zero-padding for shorter cuts."""
    audio, audio_lens = collate_audio(cuts.resample(SR))

    assert audio.shape == (2, 32000)
    assert audio_lens.tolist() == [16000, 32000]
    # Padding region for the shorter cut must be zero
    assert (audio[0, 16000:] == 0).all(), "Audio padding should be zero"
    # Non-padding region should have non-zero data (random audio from dummy_recording)
    assert (audio[0, :16000] != 0).any(), "Audio data should be non-zero"
    assert (audio[1, :32000] != 0).any(), "Audio data should be non-zero"


def test_collate_token_channel_target(cuts, tokenizer):
    """Test collate_token_channel for target (assistant) role: BOS/EOS placement, token decode."""
    pad = get_pad_id(tokenizer)
    bos = tokenizer.bos
    eos = tokenizer.eos
    total1 = compute_num_frames(1.0, FL, SR)  # 13
    total2 = compute_num_frames(2.0, FL, SR)  # 25

    target_tokens, target_token_lens = collate_token_channel(
        cuts,
        tokenizer,
        frame_length=FL,
        roles={"assistant"},
        bos_id=bos,
        eos_id=eos,
        remove_timestamps=True,
    )

    assert target_token_lens.tolist() == [total1, total2]

    # fmt: off
    # Cut 1: "hello"(22172) at frames 5–9, padded to 25
    # Cut 2: "good morning to you"(1781,7250,304,366) at frames 8–14, "welcome"(12853) at frames 20–24
    expected_target = torch.tensor([
        [0, 0, 0, 0, 0, 1, 22172, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1781, 7250, 304, 366, 0, 2, 0, 0, 0, 0, 0, 1, 12853, 0, 0, 0],
    ])
    # fmt: on
    assert torch.equal(target_tokens, expected_target)

    # Decode verification
    _verify_supervision_tokens(target_tokens[0], 0.4, 0.3, "hello", tokenizer, pad, bos, eos, total1)
    _verify_supervision_tokens(
        target_tokens[1],
        0.6,
        0.5,
        "<|0|> good <|2|> <|2|> morning <|4|> <|4|> to <|5|> <|5|> you <|6|>",
        tokenizer,
        pad,
        bos,
        eos,
        total2,
    )
    _verify_supervision_tokens(target_tokens[1], 1.6, 0.4, "<|0|> welcome <|4|>", tokenizer, pad, bos, eos, total2)


def test_collate_token_channel_source(cuts, tokenizer):
    """Test collate_token_channel for source (user) role with timestamp-based word alignment."""
    pad = get_pad_id(tokenizer)
    bos = tokenizer.bos
    eos = tokenizer.eos
    total1 = compute_num_frames(1.0, FL, SR)  # 13
    total2 = compute_num_frames(2.0, FL, SR)  # 25

    source_tokens, source_token_lens = collate_token_channel(
        cuts,
        tokenizer,
        frame_length=FL,
        roles={"user"},
        bos_id=bos,
        eos_id=eos,
        remove_timestamps=False,
        prepend_word_space=False,
    )

    assert source_tokens.shape == (2, total2)
    assert source_token_lens.tolist() == [total1, total2]

    # fmt: off
    # Cut 1: "hi"(7251) plain text, placed contiguously at frames 0–4
    # Cut 2: "good"(1781) at ts 0–1, pad gap, "morning"(7250) at ts 3–5 → frames 0–6
    #         "thanks"(3969) at ts 0–3 → frames 15–19
    expected_source = torch.tensor([
        [1, 7251, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1781, 0, 0, 7250, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3969, 0, 0, 2, 0, 0, 0, 0, 0],
    ])
    # fmt: on
    assert torch.equal(source_tokens, expected_source)

    # Decode verification
    # ── Cut 1: no timestamps, sentence-level tokenization ──
    _verify_supervision_tokens(source_tokens[0], 0.0, 0.3, "hi", tokenizer, pad, bos, eos, total1)
    assert (source_tokens[0, total1:] == pad).all(), "Batch padding should be pad"

    # ── Cut 2: two user turns with timestamp alignment ──
    _verify_supervision_tokens(
        source_tokens[1],
        0.0,
        0.5,
        "<|0|> good <|1|> <|3|> morning <|5|>",
        tokenizer,
        pad,
        bos,
        eos,
        total2,
    )
    _verify_supervision_tokens(source_tokens[1], 1.2, 0.3, "<|0|> thanks <|3|>", tokenizer, pad, bos, eos, total2)


def test_collate_system_prompt(cuts, tokenizer):
    """Test collate_system_prompt: cut1 has no prompt, cut2 has 'be helpful'."""

    prompt_tokens, prompt_token_lens = collate_system_prompt(cuts, tokenizer)

    # fmt: off
    # cut1: no system_prompt → all pad, len=0
    # cut2: "be helpful" → [BOS=1, "be"=367, "helpful"=8444, EOS=2], len=4
    expected_prompt = torch.tensor([
        [0, 0, 0, 0],
        [1, 367, 8444, 2],
    ])
    # fmt: on
    assert prompt_token_lens.tolist() == [0, 4]
    assert torch.equal(prompt_tokens, expected_prompt)

    # Decode prompt tokens back to text
    prompt_ids = tokenizer.text_to_ids("be helpful")
    decoded_prompt = tokenizer.ids_to_text(prompt_ids).strip()
    assert decoded_prompt == "be helpful", f"Prompt decode: '{decoded_prompt}'"


def test_collate_text_data(tokenizer):
    """Test collate_vectors for text token inputs: padding and lengths."""
    pad = get_pad_id(tokenizer)

    # "hi" → [7251], "good morning" → [1781, 7250]
    text_tokens_list = [
        torch.tensor([7251], dtype=torch.long),
        torch.tensor([1781, 7250], dtype=torch.long),
    ]
    text_token_lens = torch.tensor([t.shape[0] for t in text_tokens_list], dtype=torch.long)
    text_tokens = collate_vectors(text_tokens_list, padding_value=pad)

    assert text_token_lens.tolist() == [1, 2]
    assert text_tokens.shape == (2, 2)
    # Shorter sequence is right-padded
    assert text_tokens[0].tolist() == [7251, pad]
    assert text_tokens[1].tolist() == [1781, 7250]
    # Decode back to verify
    assert tokenizer.ids_to_text([7251]).strip() == "hi"
    assert tokenizer.ids_to_text([1781, 7250]).strip() == "good morning"


def test_duplex_stt_dataset(cuts, tokenizer):
    """End-to-end test of DuplexSTTDataset.__getitem__: covers all collate outputs including timestamps."""
    dataset = DuplexSTTDataset(
        tokenizer=tokenizer,
        frame_length=FL,
        source_sample_rate=SR,
        input_roles=["user"],
        output_roles=["assistant"],
        cfg={"prepend_word_space": False},
        model_cfg={"predict_user_text": True},
    )
    batch = dataset[cuts]
    ad = batch["audio_data"]

    total1 = compute_num_frames(1.0, FL, SR)  # 13
    total2 = compute_num_frames(2.0, FL, SR)  # 25

    # sample_id
    assert len(ad["sample_id"]) == 2

    # source_audio
    assert ad["source_audio"].shape == (2, 32000)
    assert ad["source_audio_lens"].tolist() == [16000, 32000]

    # target_tokens (remove_timestamps=True, same as unit test)
    assert ad["target_token_lens"].tolist() == [total1, total2]
    # fmt: off
    assert torch.equal(ad["target_tokens"], torch.tensor([
        [0, 0, 0, 0, 0, 1, 22172, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1781, 7250, 304, 366, 0, 2, 0, 0, 0, 0, 0, 1, 12853, 0, 0, 0],
    ]))
    # fmt: on

    # source_tokens (remove_timestamps=False via predict_user_text=True, timestamp-aligned)
    assert ad["source_token_lens"].tolist() == [total1, total2]
    # fmt: off
    assert torch.equal(ad["source_tokens"], torch.tensor([
        [1, 7251, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1781, 0, 0, 7250, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3969, 0, 0, 2, 0, 0, 0, 0, 0],
    ]))
    # fmt: on

    # system prompt (cut2 has "be helpful")
    assert "prompt_tokens" in ad
    assert ad["prompt_token_lens"].tolist() == [0, 4]
    # fmt: off
    assert torch.equal(ad["prompt_tokens"], torch.tensor([
        [0, 0, 0, 0],
        [1, 367, 8444, 2],
    ]))
    # fmt: on

    # source/target texts
    assert ad["source_texts"] == ["hi", "good morning thanks"]
    assert ad["target_texts"] == [
        "hello",
        "<|0|> good <|2|> <|2|> morning <|4|> <|4|> to <|5|> <|5|> you <|6|> <|0|> welcome <|4|>",
    ]

    # task
    assert ad["task"] == ["s2s_duplex", "s2s_duplex"]

    # no text data (no Formattable cuts)
    assert batch["text_data"] is None

    # source_audio is not augmented when augmenter is not configured (audio is unchanged from collate_audio)


def test_duplex_stt_dataset_augmentation(cuts, tokenizer, tmp_path):
    """Test that audio augmentation modifies source_audio in-place when configured."""
    import numpy as np
    import soundfile as sf

    # Create dummy noise files for the augmenter
    noise_dir = tmp_path / "noise" / "all"
    noise_dir.mkdir(parents=True)
    for i in range(3):
        noise = np.random.randn(SR).astype(np.float32) * 0.01
        sf.write(str(noise_dir / f"noise_{i}.wav"), noise, SR)

    cfg = {
        "prepend_word_space": False,
        "use_noise_aug": True,
        "noise_prob": 1.0,
        "noise_aug_path": str(tmp_path / "noise"),
        "noise_min_snr": 20,
        "noise_max_snr": 20,
    }

    dataset = DuplexSTTDataset(
        tokenizer=tokenizer,
        frame_length=FL,
        source_sample_rate=SR,
        input_roles=["user"],
        output_roles=["assistant"],
        cfg=cfg,
        model_cfg={"predict_user_text": True},
    )

    assert dataset.audio_augmenter is not None

    # Get original audio for comparison
    original_audio, _ = collate_audio(cuts.resample(SR))

    batch = dataset[cuts]
    ad = batch["audio_data"]

    # source_audio should be augmented (different from original)
    assert "source_audio_aug" not in ad
    assert ad["source_audio"].shape == original_audio.shape
    assert not torch.equal(ad["source_audio"], original_audio)
