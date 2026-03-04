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

from nemo.collections.asr.modules.rnnt import RNNTAttJoint
from nemo.collections.asr.parts.utils.chunking_utils import chunk_concat_audio


@pytest.fixture()
def joint_config():
    return {
        'encoder_hidden': 64,
        'pred_hidden': 64,
        'joint_hidden': 32,
        'activation': 'relu',
    }


def make_joint(joint_config, chunk_size, num_classes=10):
    return RNNTAttJoint(
        jointnet=joint_config,
        num_classes=num_classes,
        chunk_size=chunk_size,
    )


class TestChunkConcatAudio:
    """Tests for chunk_concat_audio with various boundary conditions."""

    def test_exact_division(self):
        """T is exactly divisible by chunk_size — no padding needed."""
        B, T, D = 2, 8, 4
        chunk_size = 4
        signal = torch.randn(B, T, D)
        lengths = torch.tensor([8, 6])

        chunked, sizes = chunk_concat_audio(signal, lengths, chunk_size)

        assert chunked.shape == (B, 2, chunk_size * D)
        assert sizes.shape == (B, 2)
        # First utterance: both chunks fully valid
        assert sizes[0, 0].item() == 4
        assert sizes[0, 1].item() == 4
        # Second utterance: first chunk full, second chunk partial
        assert sizes[1, 0].item() == 4
        assert sizes[1, 1].item() == 2

    def test_incomplete_last_chunk(self):
        """T is not divisible by chunk_size — last chunk is partial."""
        B, T, D = 2, 7, 4
        chunk_size = 3
        signal = torch.randn(B, T, D)
        lengths = torch.tensor([7, 5])

        chunked, sizes = chunk_concat_audio(signal, lengths, chunk_size)

        num_chunks = 3  # ceil(7/3)
        assert chunked.shape == (B, num_chunks, chunk_size * D)
        # First utterance: 3 + 3 + 1 valid frames
        assert sizes[0].tolist() == [3, 3, 1]
        # Second utterance: 3 + 2 + 0 valid frames
        assert sizes[1].tolist() == [3, 2, 0]

    def test_single_frame(self):
        """Edge case: T=1, chunk_size > 1."""
        B, T, D = 1, 1, 8
        chunk_size = 4
        signal = torch.randn(B, T, D)
        lengths = torch.tensor([1])

        chunked, sizes = chunk_concat_audio(signal, lengths, chunk_size)

        assert chunked.shape == (B, 1, chunk_size * D)
        assert sizes[0, 0].item() == 1

    def test_chunk_size_equals_one(self):
        """chunk_size=1 means each frame is its own chunk."""
        B, T, D = 2, 5, 4
        chunk_size = 1
        signal = torch.randn(B, T, D)
        lengths = torch.tensor([5, 3])

        chunked, sizes = chunk_concat_audio(signal, lengths, chunk_size)

        assert chunked.shape == (B, T, D)
        assert sizes[0].tolist() == [1, 1, 1, 1, 1]
        assert sizes[1].tolist() == [1, 1, 1, 0, 0]

    def test_chunk_size_larger_than_T(self):
        """chunk_size > T — everything fits in one chunk."""
        B, T, D = 2, 3, 4
        chunk_size = 10
        signal = torch.randn(B, T, D)
        lengths = torch.tensor([3, 2])

        chunked, sizes = chunk_concat_audio(signal, lengths, chunk_size)

        assert chunked.shape == (B, 1, chunk_size * D)
        assert sizes[0, 0].item() == 3
        assert sizes[1, 0].item() == 2

    def test_zero_length_utterance(self):
        """An utterance with length 0 should produce all-zero chunk sizes."""
        B, T, D = 2, 4, 4
        chunk_size = 2
        signal = torch.randn(B, T, D)
        lengths = torch.tensor([4, 0])

        chunked, sizes = chunk_concat_audio(signal, lengths, chunk_size)

        assert sizes[0].tolist() == [2, 2]
        assert sizes[1].tolist() == [0, 0]

    def test_padded_region_is_zero(self):
        """Padded frames beyond T should be zero in the output."""
        B, T, D = 1, 3, 2
        chunk_size = 4
        signal = torch.ones(B, T, D)
        lengths = torch.tensor([3])

        chunked, sizes = chunk_concat_audio(signal, lengths, chunk_size)

        # chunked shape: [1, 1, 8] — 4 frames * 2 features
        # First 3 frames (6 values) should be 1.0, last frame (2 values) should be 0.0
        assert chunked[0, 0, :6].sum().item() == 6.0
        assert chunked[0, 0, 6:].sum().item() == 0.0

    def test_batch_independence(self):
        """Different utterance lengths in a batch produce correct per-utterance sizes."""
        B, T, D = 4, 10, 2
        chunk_size = 3
        signal = torch.randn(B, T, D)
        lengths = torch.tensor([10, 7, 3, 1])

        _, sizes = chunk_concat_audio(signal, lengths, chunk_size)

        num_chunks = 4  # ceil(10/3)
        assert sizes.shape == (B, num_chunks)
        assert sizes[0].tolist() == [3, 3, 3, 1]
        assert sizes[1].tolist() == [3, 3, 1, 0]
        assert sizes[2].tolist() == [3, 0, 0, 0]
        assert sizes[3].tolist() == [1, 0, 0, 0]


class TestRNNTAttJoint:
    """Tests for the RNNTAttJoint module."""

    def test_joint_forward_shapes(self, joint_config):
        """Test that forward produces correct output shapes."""
        chunk_size = 2
        num_classes = 10
        joint = make_joint(joint_config, chunk_size, num_classes)

        B, T, D = 2, 6, joint_config['encoder_hidden']
        U = 4
        enc = torch.randn(B, D, T)  # (B, D, T) — channel-first
        dec = torch.randn(B, joint_config['pred_hidden'], U)
        enc_len = torch.tensor([6, 4])

        out = joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=enc_len)

        num_chunks = 3  # ceil(6/2)
        V = num_classes + 1  # +1 for blank
        assert out.shape == (B, num_chunks, U, V)

    def test_joint_method_chunking(self, joint_config):
        """Test that joint() correctly chunks when given 1D encoder lengths."""
        chunk_size = 3
        joint = make_joint(joint_config, chunk_size)

        B, T, D = 2, 7, joint_config['encoder_hidden']
        U = 3
        enc = torch.randn(B, T, D)
        dec = torch.randn(B, U, joint_config['pred_hidden'])
        enc_len = torch.tensor([7, 5])

        out = joint.joint(enc, dec, enc_len)

        num_chunks = 3  # ceil(7/3)
        assert out.shape[0] == B
        assert out.shape[1] == num_chunks
        assert out.shape[2] == U

        # num_chunks_per_utterance should be set
        assert joint.num_chunks_per_utterance is not None
        assert joint.num_chunks_per_utterance[0].item() == 3  # all 3 chunks valid
        assert joint.num_chunks_per_utterance[1].item() == 2  # only 2 chunks valid (5/3 = 1.67)

    def test_joint_method_passthrough_2d(self, joint_config):
        """Test that joint() passes through when given 2D chunk_frame_lengths."""
        chunk_size = 2
        joint = make_joint(joint_config, chunk_size)

        B, num_chunks = 2, 4
        D = chunk_size * joint_config['encoder_hidden']
        U = 3
        enc = torch.randn(B, num_chunks, D)
        dec = torch.randn(B, U, joint_config['pred_hidden'])
        chunk_frame_lengths = torch.tensor([[2, 2, 2, 1], [2, 2, 0, 0]])

        out = joint.joint(enc, dec, chunk_frame_lengths)

        assert out.shape[0] == B
        assert out.shape[1] == num_chunks
        assert out.shape[2] == U
        # Should not set num_chunks_per_utterance for 2D input
        assert joint.num_chunks_per_utterance is None

    def test_chunk_encoder_for_decoding(self, joint_config):
        """Test chunk_encoder_for_decoding produces correct shapes and values."""
        chunk_size = 3
        joint = make_joint(joint_config, chunk_size)

        B, T, D = 2, 7, joint_config['encoder_hidden']
        enc = torch.randn(B, D, T)  # channel-first as from encoder
        enc_len = torch.tensor([7, 4])

        chunked_enc, num_chunks, chunk_lengths = joint.chunk_encoder_for_decoding(enc, enc_len)

        expected_num_chunks_total = 3  # ceil(7/3)
        assert chunked_enc.shape == (B, chunk_size * D, expected_num_chunks_total)
        assert num_chunks[0].item() == 3  # 7 frames -> 3 valid chunks
        assert num_chunks[1].item() == 2  # 4 frames -> 2 valid chunks
        assert chunk_lengths.shape == (B, expected_num_chunks_total)
        assert chunk_lengths[0].tolist() == [3, 3, 1]
        assert chunk_lengths[1].tolist() == [3, 1, 0]

    def test_incomplete_chunk_masking(self, joint_config):
        """
        Verify that cross-attention masking works for incomplete chunks.
        With an incomplete last chunk, the padded frames should be masked out
        and not affect the output significantly compared to full chunks.
        """
        chunk_size = 4
        joint = make_joint(joint_config, chunk_size)
        joint.eval()

        B, D = 1, joint_config['encoder_hidden']
        U = 2

        # Create two scenarios: T=4 (exact) vs T=5 (one extra frame -> incomplete 2nd chunk)
        enc_exact = torch.randn(B, D, 4)
        enc_extra = torch.cat([enc_exact, torch.randn(B, D, 1)], dim=2)
        dec = torch.randn(B, joint_config['pred_hidden'], U)

        with torch.no_grad():
            out_exact = joint(
                encoder_outputs=enc_exact, decoder_outputs=dec,
                encoder_lengths=torch.tensor([4])
            )
            out_extra = joint(
                encoder_outputs=enc_extra, decoder_outputs=dec,
                encoder_lengths=torch.tensor([5])
            )

        # Exact: 1 chunk. Extra: 2 chunks (second has only 1 valid frame out of 4).
        assert out_exact.shape[1] == 1
        assert out_extra.shape[1] == 2
        # The first chunk output should be identical since input is the same
        torch.testing.assert_close(out_exact[:, 0, :, :], out_extra[:, 0, :, :])

    def test_gradient_flow(self, joint_config):
        """Verify gradients flow through the joint for training."""
        chunk_size = 2
        joint = make_joint(joint_config, chunk_size)

        B, T, D = 2, 5, joint_config['encoder_hidden']
        U = 3
        enc = torch.randn(B, D, T, requires_grad=True)
        dec = torch.randn(B, joint_config['pred_hidden'], U)
        enc_len = torch.tensor([5, 3])

        out = joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=enc_len)
        loss = out.sum()
        loss.backward()

        assert enc.grad is not None
        assert enc.grad.shape == enc.shape
