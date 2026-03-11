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

import pytest
import torch

from nemo.collections.asr.parts.rnnt_triton.rnnt_align import align_from_logits, rnnt_best_path_align
from nemo.core.utils.optional_libs import K2_AVAILABLE

from tests.collections.asr.decoding.utils import avoid_sync_operations

if K2_AVAILABLE:
    from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss


def get_devices_for_testing():
    """Return list of available devices for testing, preferring CUDA > MPS > CPU."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.insert(0, torch.device("cuda"))
    elif torch.backends.mps.is_available():
        devices.insert(0, torch.device("mps"))
    return devices


def get_devices_for_k2_testing():
    """Return list of devices supported by k2 (CPU and CUDA only, not MPS)."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.insert(0, torch.device("cuda"))
    return devices


DEVICES = get_devices_for_testing()
DEVICES_FOR_K2 = get_devices_for_k2_testing()


class TestRnntBestPathAlign:
    """Unit tests for rnnt_best_path_align function."""

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_no_cuda_sync_operations(self):
        """Verify no CUDA-CPU synchronization during alignment."""
        batch_size, src_length, tgt_length, vocab_size = 4, 100, 20, 32
        device = torch.device("cuda")

        target_logprobs = torch.randn([batch_size, src_length, tgt_length + 1], device=device)
        blank_logprobs = torch.randn([batch_size, src_length, tgt_length + 1], device=device)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        with avoid_sync_operations(device):
            alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        assert alignments.shape == (batch_size, tgt_length)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_all_symbols_at_frame_0(self, device):
        """Test case where all symbols should be emitted at frame 0 (very high emit scores)."""
        B, T, U = 1, 4, 3
        dtype = torch.float32

        # Create logprobs where emit scores are very high at frame 0
        target_logprobs = torch.full([B, T, U + 1], -10.0, dtype=dtype, device=device)
        blank_logprobs = torch.full([B, T, U + 1], -10.0, dtype=dtype, device=device)

        # Make emit very favorable at frame 0 for all positions
        target_logprobs[0, 0, :U] = 0.0  # High emit probability at frame 0

        # Make blank favorable after emitting all symbols
        blank_logprobs[0, :, U] = 0.0  # After emitting all, blank is favorable

        src_lengths = torch.tensor([T], device=device)
        tgt_lengths = torch.tensor([U], device=device)

        alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        # All symbols should be emitted at frame 0
        expected = torch.tensor([[0, 0, 0]], dtype=torch.long, device=device)
        assert torch.equal(alignments, expected), f"Expected {expected}, got {alignments}"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_all_symbols_at_last_frame(self, device):
        """Test case where all symbols should be emitted at the last frame."""
        B, T, U = 1, 4, 3
        dtype = torch.float32

        # Create logprobs where blank is favorable until the last frame
        target_logprobs = torch.full([B, T, U + 1], -10.0, dtype=dtype, device=device)
        blank_logprobs = torch.full([B, T, U + 1], 0.0, dtype=dtype, device=device)

        # Make emit very favorable at the last frame (T-1)
        target_logprobs[0, T - 1, :U] = 0.0

        src_lengths = torch.tensor([T], device=device)
        tgt_lengths = torch.tensor([U], device=device)

        alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        # All symbols should be emitted at the last frame (T-1)
        expected = torch.tensor([[T - 1, T - 1, T - 1]], dtype=torch.long, device=device)
        assert torch.equal(alignments, expected), f"Expected {expected}, got {alignments}"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_one_symbol_per_frame(self, device):
        """Test case where one symbol is emitted per frame."""
        B, T, U = 1, 3, 3
        dtype = torch.float32

        # Create logprobs to encourage one symbol per frame
        target_logprobs = torch.full([B, T, U + 1], -10.0, dtype=dtype, device=device)
        blank_logprobs = torch.full([B, T, U + 1], -10.0, dtype=dtype, device=device)

        # Emit symbol 0 at frame 0, symbol 1 at frame 1, symbol 2 at frame 2
        target_logprobs[0, 0, 0] = 0.0  # emit u=0 at t=0
        target_logprobs[0, 1, 1] = 0.0  # emit u=1 at t=1
        target_logprobs[0, 2, 2] = 0.0  # emit u=2 at t=2

        src_lengths = torch.tensor([T], device=device)
        tgt_lengths = torch.tensor([U], device=device)

        alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        expected = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        assert torch.equal(alignments, expected), f"Expected {expected}, got {alignments}"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_T_equals_1(self, device):
        """Test edge case where T=1: all tokens must be emitted at frame 0."""
        B, T, U = 1, 1, 3
        dtype = torch.float32

        target_logprobs = torch.zeros([B, T, U + 1], dtype=dtype, device=device)
        blank_logprobs = torch.zeros([B, T, U + 1], dtype=dtype, device=device)

        src_lengths = torch.tensor([T], device=device)
        tgt_lengths = torch.tensor([U], device=device)

        alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        # All tokens must be at frame 0 since T=1
        expected = torch.tensor([[0, 0, 0]], dtype=torch.long, device=device)
        assert torch.equal(alignments, expected), f"Expected {expected}, got {alignments}"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_U_equals_0(self, device):
        """Test edge case where U=0: nothing to align."""
        B, T, U = 2, 5, 0
        dtype = torch.float32

        target_logprobs = torch.zeros([B, T, U + 1], dtype=dtype, device=device)
        blank_logprobs = torch.zeros([B, T, U + 1], dtype=dtype, device=device)

        src_lengths = torch.tensor([T, T], device=device)
        tgt_lengths = torch.tensor([0, 0], device=device)

        alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        assert alignments.shape == (B, 0), f"Expected shape (2, 0), got {alignments.shape}"

    @pytest.mark.unit
    def test_T_equals_0_and_U_greater_than_0_raises(self):
        """Test that T=0 with U>0 raises an error (impossible alignment).

        Note: This validation only works on CPU. On GPU, the check is skipped to avoid
        CUDA-CPU synchronization, and invalid results will be produced silently.
        """
        B, T, U = 1, 5, 3
        dtype = torch.float32
        device = torch.device("cpu")

        target_logprobs = torch.zeros([B, T, U + 1], dtype=dtype, device=device)
        blank_logprobs = torch.zeros([B, T, U + 1], dtype=dtype, device=device)

        src_lengths = torch.tensor([0], device=device)  # T=0
        tgt_lengths = torch.tensor([U], device=device)  # U>0

        with pytest.raises(ValueError, match="Cannot align sequences with T=0 and U>0"):
            rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_variable_lengths_in_batch(self, device):
        """Test batch with variable source and target lengths."""
        B, T_max, U_max = 3, 5, 4
        dtype = torch.float32

        target_logprobs = torch.zeros([B, T_max, U_max + 1], dtype=dtype, device=device)
        blank_logprobs = torch.zeros([B, T_max, U_max + 1], dtype=dtype, device=device)

        # Make emit very favorable everywhere
        target_logprobs.fill_(-1.0)
        blank_logprobs.fill_(-10.0)  # Discourage blanks

        src_lengths = torch.tensor([5, 3, 4], device=device)
        tgt_lengths = torch.tensor([4, 2, 3], device=device)

        alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        assert alignments.shape == (B, U_max)

        # Check that alignments are within valid range for each batch
        for b in range(B):
            T_b = src_lengths[b].item()
            U_b = tgt_lengths[b].item()
            for u in range(U_b):
                assert 0 <= alignments[b, u].item() < T_b, f"Alignment out of range for batch {b}, position {u}"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_hand_crafted_small_lattice(self, device):
        """Test with a small hand-crafted lattice (T=3, U=2) with known best path."""
        B, T, U = 1, 3, 2
        dtype = torch.float32

        # Design logprobs for a specific best path
        # We want the path: emit at t=0 (u=0->1), blank at t=0->1, emit at t=1 (u=1->2), blank at t=1->2, blank at t=2->3
        # So alignment should be [0, 1]

        target_logprobs = torch.full([B, T, U + 1], -5.0, dtype=dtype, device=device)
        blank_logprobs = torch.full([B, T, U + 1], -1.0, dtype=dtype, device=device)

        # Make emit favorable at specific positions
        target_logprobs[0, 0, 0] = 0.0  # emit u=0 at t=0 (very favorable)
        target_logprobs[0, 1, 1] = 0.0  # emit u=1 at t=1 (very favorable)

        src_lengths = torch.tensor([T], device=device)
        tgt_lengths = torch.tensor([U], device=device)

        alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        expected = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        assert torch.equal(alignments, expected), f"Expected {expected}, got {alignments}"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_batch_size_1(self, device):
        """Test with single batch element."""
        B, T, U = 1, 4, 2
        dtype = torch.float32

        target_logprobs = torch.randn([B, T, U + 1], dtype=dtype, device=device)
        blank_logprobs = torch.randn([B, T, U + 1], dtype=dtype, device=device)

        src_lengths = torch.tensor([T], device=device)
        tgt_lengths = torch.tensor([U], device=device)

        alignments = rnnt_best_path_align(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)

        assert alignments.shape == (B, U)
        # Check monotonicity
        for u in range(U - 1):
            assert alignments[0, u] <= alignments[0, u + 1], "Alignments should be monotonically non-decreasing"


class TestAlignFromLogits:
    """Unit tests for align_from_logits function."""

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_basic_alignment(self, device):
        """Test basic alignment from logits."""
        B, T, U, V = 2, 4, 2, 5
        blank_id = 0

        logits = torch.randn([B, T, U + 1, V], device=device)
        targets = torch.randint(1, V, [B, U], device=device)  # Avoid blank in targets
        src_lengths = torch.tensor([T, T], device=device)
        tgt_lengths = torch.tensor([U, U], device=device)

        alignments = align_from_logits(logits, targets, blank_id, src_lengths, tgt_lengths)

        assert alignments.shape == (B, U)
        # Check alignments are within valid range
        for b in range(B):
            for u in range(U):
                assert 0 <= alignments[b, u].item() < T

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_alignment_without_lengths(self, device):
        """Test alignment when lengths are not provided (use max lengths)."""
        B, T, U, V = 1, 3, 2, 4
        blank_id = V - 1  # Blank at end

        logits = torch.randn([B, T, U + 1, V], device=device)
        targets = torch.randint(0, V - 1, [B, U], device=device)

        alignments = align_from_logits(logits, targets, blank_id)

        assert alignments.shape == (B, U)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_alignment_monotonicity(self, device):
        """Test that alignments are monotonically non-decreasing."""
        B, T, U, V = 4, 10, 5, 8
        blank_id = 0

        torch.manual_seed(42)
        logits = torch.randn([B, T, U + 1, V], device=device)
        targets = torch.randint(1, V, [B, U], device=device)
        src_lengths = torch.full([B], T, device=device, dtype=torch.long)
        tgt_lengths = torch.full([B], U, device=device, dtype=torch.long)

        alignments = align_from_logits(logits, targets, blank_id, src_lengths, tgt_lengths)

        for b in range(B):
            for u in range(U - 1):
                assert (
                    alignments[b, u] <= alignments[b, u + 1]
                ), f"Alignment not monotonic at batch {b}: {alignments[b].tolist()}"


@pytest.mark.skipif(not K2_AVAILABLE, reason="k2 not available")
class TestAgainstK2Reference:
    """Tests comparing against k2 reference implementation."""

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES_FOR_K2)
    @pytest.mark.parametrize(
        "B,T,U,V",
        [
            (1, 4, 2, 4),
            (2, 5, 3, 6),
            (4, 10, 5, 32),
        ],
    )
    def test_matches_k2_random(self, B, T, U, V, device):
        """Compare alignment against k2 GraphRnntLoss.align_from_logits."""
        blank_id = 0
        torch.manual_seed(123)

        logits = torch.randn([B, T, U + 1, V], device=device)
        targets = torch.randint(1, V, [B, U], device=device)
        src_lengths = torch.full([B], T, device=device, dtype=torch.long)
        tgt_lengths = torch.full([B], U, device=device, dtype=torch.long)

        # Our implementation
        our_alignments = align_from_logits(logits, targets, blank_id, src_lengths, tgt_lengths)

        # k2 reference
        k2_loss = GraphRnntLoss(blank=blank_id)
        k2_alignments = k2_loss.align_from_logits(logits, targets, src_lengths, tgt_lengths)

        assert torch.equal(
            our_alignments, k2_alignments
        ), f"Alignment mismatch:\nOurs: {our_alignments}\nK2: {k2_alignments}"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES_FOR_K2)
    @pytest.mark.parametrize(
        "B,T,U,V",
        [
            (1, 4, 2, 4),
            (4, 10, 5, 32),
        ],
    )
    def test_matches_k2_blank_last(self, B, T, U, V, device):
        """Compare alignment with blank_id at the end of vocabulary."""
        blank_id = V - 1
        torch.manual_seed(456)

        logits = torch.randn([B, T, U + 1, V], device=device)
        targets = torch.randint(0, V - 1, [B, U], device=device)  # Avoid blank
        src_lengths = torch.full([B], T, device=device, dtype=torch.long)
        tgt_lengths = torch.full([B], U, device=device, dtype=torch.long)

        our_alignments = align_from_logits(logits, targets, blank_id, src_lengths, tgt_lengths)

        k2_loss = GraphRnntLoss(blank=blank_id)
        k2_alignments = k2_loss.align_from_logits(logits, targets, src_lengths, tgt_lengths)

        assert torch.equal(
            our_alignments, k2_alignments
        ), f"Alignment mismatch:\nOurs: {our_alignments}\nK2: {k2_alignments}"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES_FOR_K2)
    def test_matches_k2_variable_lengths(self, device):
        """Compare alignment with variable lengths in batch."""
        B, T_max, U_max, V = 4, 12, 8, 16
        blank_id = 0
        torch.manual_seed(789)

        logits = torch.randn([B, T_max, U_max + 1, V], device=device)
        targets = torch.randint(1, V, [B, U_max], device=device)
        src_lengths = torch.tensor([12, 10, 8, 6], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([8, 6, 5, 3], device=device, dtype=torch.long)

        our_alignments = align_from_logits(logits, targets, blank_id, src_lengths, tgt_lengths)

        k2_loss = GraphRnntLoss(blank=blank_id)
        k2_alignments = k2_loss.align_from_logits(logits, targets, src_lengths, tgt_lengths)

        # Compare only valid positions
        for b in range(B):
            U_b = tgt_lengths[b].item()
            assert torch.equal(
                our_alignments[b, :U_b], k2_alignments[b, :U_b]
            ), f"Alignment mismatch at batch {b}:\nOurs: {our_alignments[b, :U_b]}\nK2: {k2_alignments[b, :U_b]}"


class TestWithSampleData:
    """Tests using sample data from conftest fixtures."""

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_sample_small(self, device, rnn_loss_sample_data):
        """Test alignment with small sample data."""
        sample = rnn_loss_sample_data.get_sample_small()
        logits = sample.logits.to(device)
        targets = sample.targets.to(device)
        src_lengths = sample.input_lengths.to(device)
        tgt_lengths = sample.target_lengths.to(device)
        blank_id = sample.blank_id

        alignments = align_from_logits(logits, targets, blank_id, src_lengths, tgt_lengths)

        B = logits.shape[0]
        U = targets.shape[1]
        assert alignments.shape == (B, U)

        # Check monotonicity and bounds
        for b in range(B):
            T_b = src_lengths[b].item()
            U_b = tgt_lengths[b].item()
            for u in range(U_b):
                assert 0 <= alignments[b, u].item() < T_b

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_sample_medium(self, device, rnn_loss_sample_data):
        """Test alignment with medium sample data (batch size 2)."""
        sample = rnn_loss_sample_data.get_sample_medium()
        logits = sample.logits.to(device)
        targets = sample.targets.to(device)
        src_lengths = sample.input_lengths.to(device)
        tgt_lengths = sample.target_lengths.to(device)
        blank_id = sample.blank_id

        alignments = align_from_logits(logits, targets, blank_id, src_lengths, tgt_lengths)

        B = logits.shape[0]
        U = targets.shape[1]
        assert alignments.shape == (B, U)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_sample_small_blank_last(self, device, rnn_loss_sample_data):
        """Test alignment with blank at end of vocabulary."""
        sample = rnn_loss_sample_data.get_sample_small_blank_last()
        logits = sample.logits.to(device)
        targets = sample.targets.to(device)
        src_lengths = sample.input_lengths.to(device)
        tgt_lengths = sample.target_lengths.to(device)
        blank_id = sample.blank_id

        alignments = align_from_logits(logits, targets, blank_id, src_lengths, tgt_lengths)

        B = logits.shape[0]
        U = targets.shape[1]
        assert alignments.shape == (B, U)


@pytest.mark.skipif(not K2_AVAILABLE, reason="k2 not available")
class TestRealModelAlignment:
    """Integration tests with real ASR models."""

    @pytest.mark.unit
    @pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.mps.is_available()), reason="Requires CUDA or MPS for k2 comparison"
    )
    def test_real_model_alignment_validity(self, fast_conformer_transducer_model):
        """Test alignment validity with a real transducer model."""
        model = fast_conformer_transducer_model
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        k2_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )  # k2 does not support MPS

        model = model.to(device)
        model.eval()

        # Create some dummy input (batch of 2, ~1 second of audio at 16kHz)
        # FastConformer expects mel-spectrogram features, not raw audio
        # Use the preprocessor to get the correct input shape
        B = 2
        audio_length = 16000
        raw_audio = torch.randn(B, audio_length, device=device)
        audio_lengths = torch.tensor([audio_length, audio_length - 1000], device=device, dtype=torch.long)

        with torch.no_grad():
            # Process through preprocessor to get mel features
            processed_signal, processed_signal_length = model.preprocessor(
                input_signal=raw_audio, length=audio_lengths
            )

            # Get encoder output
            encoded, encoded_lengths = model.encoder(audio_signal=processed_signal, length=processed_signal_length)

            # Create dummy targets (use tokenizer if available, otherwise random)
            T = encoded.shape[1]
            U = min(10, T)  # Ensure U <= T is reasonable
            V = model.decoder.vocab_size

            # Get blank_id from model
            blank_id = model.joint.num_classes_with_blank - 1

            targets = torch.randint(0, blank_id, [B, U], device=device)
            tgt_lengths = torch.tensor([U, U - 2], device=device, dtype=torch.long)

            # Get predictor output
            decoder_output, _, _ = model.decoder(targets=targets, target_length=tgt_lengths)

            # Joint: [B, T, U+1, V]
            # Temporarily disable fuse_loss_wer to avoid needing to provide all loss inputs
            original_fuse_loss_wer = model.joint._fuse_loss_wer
            model.joint._fuse_loss_wer = False
            try:
                joint_output = model.joint(
                    encoder_outputs=encoded,
                    decoder_outputs=decoder_output,
                )
            finally:
                model.joint._fuse_loss_wer = original_fuse_loss_wer

            # Get alignment
            src_lengths = encoded_lengths
            our_alignments = align_from_logits(joint_output, targets, blank_id, src_lengths, tgt_lengths)

            # Compare with k2
            k2_loss = GraphRnntLoss(blank=blank_id)
            k2_alignments = k2_loss.align_from_logits(
                joint_output.to(k2_device), targets.to(k2_device), src_lengths.to(k2_device), tgt_lengths.to(k2_device)
            )
            k2_alignments = k2_alignments.to(device)

            # Check alignment properties
            for b in range(B):
                T_b = src_lengths[b].item()
                U_b = tgt_lengths[b].item()

                # Check bounds
                for u in range(U_b):
                    assert (
                        0 <= our_alignments[b, u].item() < T_b
                    ), f"Alignment out of bounds: {our_alignments[b, u]} not in [0, {T_b})"

                # Check monotonicity
                for u in range(U_b - 1):
                    assert (
                        our_alignments[b, u] <= our_alignments[b, u + 1]
                    ), f"Alignment not monotonic: {our_alignments[b].tolist()}"

                # Check matches k2
                assert torch.equal(
                    our_alignments[b, :U_b], k2_alignments[b, :U_b]
                ), f"Alignment mismatch with k2 at batch {b}"
