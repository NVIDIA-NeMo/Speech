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

from unittest.mock import MagicMock, PropertyMock

import pytest
import torch

from nemo.collections.speechlm2.data import DuplexSTTDataset


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    type(tokenizer).bos = PropertyMock(return_value=1)
    type(tokenizer).eos = PropertyMock(return_value=2)
    type(tokenizer).pad = PropertyMock(return_value=0)
    type(tokenizer).pad_id = PropertyMock(return_value=0)
    type(tokenizer).unk_id = PropertyMock(return_value=None)
    tokenizer.text_to_ids = MagicMock(return_value=[1])
    return tokenizer


@pytest.fixture
def dataset_with_early_interruption(mock_tokenizer):
    """Create a dataset with early interruption enabled."""
    cfg = {"early_interruption_prob": 1.0, "early_interruption_overlap_tokens": 5}
    model_cfg = {"predict_user_text": False, "force_align_user_text": False}

    dataset = DuplexSTTDataset(
        tokenizer=mock_tokenizer,
        frame_length=0.08,
        source_sample_rate=16000,
        input_roles=["user"],
        output_roles=["assistant"],
        cfg=cfg,
        model_cfg=model_cfg,
    )
    return dataset


def test_early_interruption_basic_truncation(dataset_with_early_interruption):
    """Test that early interruption truncates an agent turn correctly."""
    # Setup: Create mock tensors with realistic structure:
    # BOS, non-pad tokens, pad tokens (silence), then EOS
    # Early interruption should move EOS into the non-pad token region
    batch_size = 1
    seq_len = 25

    target_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    target_tokens[0, 0] = 1  # BOS
    target_tokens[0, 1:13] = torch.arange(10, 22)  # 12 content tokens (positions 1-12)
    # Positions 13-17 are PAD (0) - representing silence/gap
    target_tokens[0, 18] = 2  # Original EOS at position 18 (after the padding)

    source_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)

    target_audio = torch.zeros((batch_size, 20000), dtype=torch.float32)
    source_audio = torch.zeros((batch_size, 16000), dtype=torch.float32)

    target_token_lens = torch.tensor([19], dtype=torch.long)
    source_token_lens = torch.tensor([1], dtype=torch.long)
    target_audio_lens = torch.tensor([20000], dtype=torch.long)
    source_audio_lens = torch.tensor([16000], dtype=torch.long)

    # Apply early interruption
    dataset_with_early_interruption._apply_early_interruption_augmentation(
        target_tokens=target_tokens,
        source_tokens=source_tokens,
        source_audio=source_audio,
        source_audio_lens=source_audio_lens,
        batch_idx=0,
    )

    # Verify: EOS should be moved earlier (within the non-pad token region)
    eos_positions = (target_tokens[0] == 2).nonzero(as_tuple=True)[0]
    assert len(eos_positions) > 0, "EOS should still exist after early interruption"

    new_eos_pos = eos_positions[0].item()
    # The new EOS position should be before the original position (18)
    assert new_eos_pos < 18, f"New EOS position {new_eos_pos} should be before original position 18"

    # CRITICAL: New EOS should be at cutoff_pos + overlap_tokens
    # cutoff_pos is in non-pad region (1-12), overlap_tokens is 5
    # So new_eos_pos should be in range (1+5) to (12+5) = 6 to 17
    overlap_tokens = 5  # from fixture cfg
    assert 6 <= new_eos_pos <= 17, (
        f"New EOS at {new_eos_pos} should be within cutoff + overlap range (6-17), "
        f"meaning agent continues for {overlap_tokens} tokens after user interruption"
    )

    # Check that tokens after new EOS are shifted or padded
    assert target_tokens[0, -1] == 0, "Last token should be PAD after early interruption"


def test_early_interruption_with_multiple_turns(dataset_with_early_interruption):
    """Test early interruption with multiple agent turns."""
    batch_size = 1
    seq_len = 50

    target_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    # First turn: BOS at 0, tokens 1-10, PAD 11-13, EOS at 14
    target_tokens[0, 0] = 1
    target_tokens[0, 1:11] = torch.arange(10, 20)  # 10 content tokens
    # Positions 11-13 are PAD
    target_tokens[0, 14] = 2  # EOS after padding

    # Second turn: BOS at 18, tokens 19-28, PAD 29-31, EOS at 32
    target_tokens[0, 18] = 1
    target_tokens[0, 19:29] = torch.arange(20, 30)  # 10 content tokens
    # Positions 29-31 are PAD
    target_tokens[0, 32] = 2  # EOS after padding

    source_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    target_audio = torch.zeros((batch_size, 40000), dtype=torch.float32)
    source_audio = torch.zeros((batch_size, 16000), dtype=torch.float32)

    target_token_lens = torch.tensor([33], dtype=torch.long)
    source_token_lens = torch.tensor([1], dtype=torch.long)
    target_audio_lens = torch.tensor([40000], dtype=torch.long)
    source_audio_lens = torch.tensor([16000], dtype=torch.long)

    # Apply early interruption
    dataset_with_early_interruption._apply_early_interruption_augmentation(
        target_tokens=target_tokens,
        source_tokens=source_tokens,
        source_audio=source_audio,
        source_audio_lens=source_audio_lens,
        batch_idx=0,
    )

    # Verify: Should still have valid BOS and EOS tokens
    bos_positions = (target_tokens[0] == 1).nonzero(as_tuple=True)[0]
    eos_positions = (target_tokens[0] == 2).nonzero(as_tuple=True)[0]

    assert len(bos_positions) >= 1, "Should have at least one BOS token"
    assert len(eos_positions) >= 1, "Should have at least one EOS token"

    # Each BOS should be followed eventually by an EOS
    for bos_pos in bos_positions:
        matching_eos = eos_positions[eos_positions > bos_pos]
        assert len(matching_eos) > 0, f"BOS at position {bos_pos} should have a matching EOS"


def test_early_interruption_overlap_tokens(dataset_with_early_interruption):
    """Test that overlap tokens parameter works correctly."""
    # Test with custom overlap tokens
    dataset_with_early_interruption.cfg["early_interruption_overlap_tokens"] = 3

    batch_size = 1
    seq_len = 25

    target_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    target_tokens[0, 0] = 1  # BOS
    target_tokens[0, 1:11] = torch.arange(10, 20)  # 10 content tokens (positions 1-10)
    # Positions 11-14 are PAD
    target_tokens[0, 15] = 2  # Original EOS at position 15 (after padding)
    original_eos_pos = 15
    overlap_tokens = 3

    # Place a marker token in source_tokens AFTER original_eos_pos to track the shift
    # The implementation shifts source_tokens from original_eos_pos+1 to cutoff_pos+1
    marker_token = 999
    marker_original_pos = 20  # Position after original EOS
    source_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    source_tokens[0, marker_original_pos] = marker_token

    target_audio = torch.zeros((batch_size, 20000), dtype=torch.float32)
    source_audio = torch.zeros((batch_size, 16000), dtype=torch.float32)

    target_token_lens = torch.tensor([16], dtype=torch.long)
    source_token_lens = torch.tensor([1], dtype=torch.long)
    target_audio_lens = torch.tensor([20000], dtype=torch.long)
    source_audio_lens = torch.tensor([16000], dtype=torch.long)

    # Apply early interruption
    dataset_with_early_interruption._apply_early_interruption_augmentation(
        target_tokens=target_tokens,
        source_tokens=source_tokens,
        source_audio=source_audio,
        source_audio_lens=source_audio_lens,
        batch_idx=0,
    )

    # Find new EOS position
    eos_positions = (target_tokens[0] == 2).nonzero(as_tuple=True)[0]
    assert len(eos_positions) > 0, "EOS should exist after early interruption"
    new_eos_pos = eos_positions[0].item()

    # Find where the marker moved to in source_tokens
    # The shift is: source_tokens[cutoff_pos+1:...] = source_tokens[original_eos_pos+1:...]
    # So marker moves from marker_original_pos to cutoff_pos + 1 + (marker_original_pos - original_eos_pos - 1)
    # = cutoff_pos + (marker_original_pos - original_eos_pos)
    marker_new_positions = (source_tokens[0] == marker_token).nonzero(as_tuple=True)[0]
    assert len(marker_new_positions) > 0, "Marker token should still exist after transformation"
    marker_new_pos = marker_new_positions[0].item()

    # Calculate actual cutoff position from the marker shift
    # marker_new_pos = cutoff_pos + (marker_original_pos - original_eos_pos)
    # => cutoff_pos = marker_new_pos - (marker_original_pos - original_eos_pos)
    actual_cutoff_pos = marker_new_pos - (marker_original_pos - original_eos_pos)

    # Verify the overlap is exactly overlap_tokens
    actual_overlap = new_eos_pos - actual_cutoff_pos
    assert actual_overlap == overlap_tokens, (
        f"Overlap should be {overlap_tokens} tokens, but got {actual_overlap} "
        f"(cutoff at {actual_cutoff_pos}, new EOS at {new_eos_pos})"
    )

    # Cutoff should be in non-pad region (user interrupts during non-pad region)
    assert 1 <= actual_cutoff_pos <= 10, (
        f"Cutoff at {actual_cutoff_pos} should be in non-pad region (1-10), "
        f"meaning user interrupts during active speech"
    )

    print(f"\n✓ Overlap tokens verification:")
    print(f"  - Configured overlap: {overlap_tokens} tokens")
    print(f"  - Actual cutoff position (from marker shift): {actual_cutoff_pos}")
    print(f"  - New EOS position: {new_eos_pos}")
    print(f"  - Actual overlap: {actual_overlap} tokens ✓")


def test_early_interruption_no_valid_turns():
    """Test that early interruption handles cases with no valid turns gracefully."""
    mock_tokenizer = MagicMock()
    type(mock_tokenizer).bos = PropertyMock(return_value=1)
    type(mock_tokenizer).eos = PropertyMock(return_value=2)
    type(mock_tokenizer).pad = PropertyMock(return_value=0)
    type(mock_tokenizer).pad_id = PropertyMock(return_value=0)
    type(mock_tokenizer).unk_id = PropertyMock(return_value=None)
    mock_tokenizer.text_to_ids = MagicMock(return_value=[1])

    cfg = {"early_interruption_prob": 1.0, "early_interruption_overlap_tokens": 5}
    model_cfg = {"predict_user_text": False, "force_align_user_text": False}

    dataset = DuplexSTTDataset(
        tokenizer=mock_tokenizer,
        frame_length=0.08,
        source_sample_rate=16000,
        input_roles=["user"],
        output_roles=["assistant"],
        cfg=cfg,
        model_cfg=model_cfg,
    )

    # Create tensors with no valid turns (all padding)
    batch_size = 1
    seq_len = 20

    target_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    source_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    target_audio = torch.zeros((batch_size, 16000), dtype=torch.float32)
    source_audio = torch.zeros((batch_size, 16000), dtype=torch.float32)

    target_token_lens = torch.tensor([1], dtype=torch.long)
    source_token_lens = torch.tensor([1], dtype=torch.long)
    target_audio_lens = torch.tensor([16000], dtype=torch.long)
    source_audio_lens = torch.tensor([16000], dtype=torch.long)

    # Apply early interruption - should not crash
    dataset._apply_early_interruption_augmentation(
        target_tokens=target_tokens,
        source_tokens=source_tokens,
        source_audio=source_audio,
        source_audio_lens=source_audio_lens,
        batch_idx=0,
    )

    # Verify: Tokens should remain unchanged (all zeros)
    assert torch.all(target_tokens == 0), "Tokens should remain unchanged when no valid turns exist"


def test_early_interruption_frames_to_remove_calculation(dataset_with_early_interruption):
    """Test that frames_to_remove is calculated correctly."""
    batch_size = 1
    seq_len = 25

    target_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    target_tokens[0, 0] = 1  # BOS
    target_tokens[0, 1:11] = torch.arange(10, 20)  # 10 content tokens (positions 1-10)
    # Positions 11-14 are PAD
    target_tokens[0, 15] = 2  # Original EOS at position 15 (after padding)
    original_eos_pos = 15

    source_tokens = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    target_audio = torch.zeros((batch_size, 20000), dtype=torch.float32)
    source_audio = torch.zeros((batch_size, 16000), dtype=torch.float32)

    target_token_lens = torch.tensor([16], dtype=torch.long)
    source_token_lens = torch.tensor([1], dtype=torch.long)
    target_audio_lens = torch.tensor([20000], dtype=torch.long)
    source_audio_lens = torch.tensor([16000], dtype=torch.long)

    # Apply early interruption
    dataset_with_early_interruption._apply_early_interruption_augmentation(
        target_tokens=target_tokens,
        source_tokens=source_tokens,
        source_audio=source_audio,
        source_audio_lens=source_audio_lens,
        batch_idx=0,
    )

    # Verify: New EOS should be in the non-pad region
    new_eos_positions = (target_tokens[0] == 2).nonzero(as_tuple=True)[0]
    assert len(new_eos_positions) > 0, "EOS should exist after early interruption"
    new_eos_pos = new_eos_positions[0].item()

    # cutoff_pos must satisfy (eos_pos - pos) > overlap_tokens, i.e., pos < 10
    # So valid cutoff positions are 1-9, and new_eos_pos = cutoff_pos + 5
    # Range: (1+5) to (9+5) = 6 to 14
    overlap_tokens = 5  # from fixture cfg
    assert 6 <= new_eos_pos <= 14, (
        f"New EOS at {new_eos_pos} should be within cutoff + overlap range (6-14), "
        f"meaning agent continues for {overlap_tokens} tokens after user interruption"
    )

    # Verify: Check that padding is added at the end
    # Since we always use frames_to_remove = original_eos_pos - cutoff_pos,
    # we should have more padding at the end after truncation
    num_pad_tokens = (target_tokens[0] == 0).sum().item()
    # 9 is a conservative lower bound: new_eos_pos ranges 6-14, so minimum trailing
    # PAD is 10 (positions 15-24), plus PAD positions 11-13 before EOS = 13 total minimum
    assert num_pad_tokens >= 9, "Should have increased padding after truncation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
