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

"""Capacity mode of the packed path: fixed-size buffers, bounded metadata, no host read, same results."""

import pytest
import torch

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.modules import ConformerEncoder
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.parts.packed_sequence import (
    PackedCapacity,
    PackedEncoderActivations,
    packed_slot_layout,
    unpack_encoder_output,
)
from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext

SAMPLE_RATE = 16_000


def _preprocessor():
    return AudioToMelSpectrogramPreprocessor(
        sample_rate=SAMPLE_RATE,
        features=16,
        n_fft=512,
        window_size=0.025,
        window_stride=0.01,
        normalize={
            "_target_": "nemo.collections.asr.parts.submodules.batchnorm.MaskedBatchNorm1d",
            "affine": False,
            "eps": 1e-3,
        },
        dither=0.0,
        pad_to=0,
    )


def _encoder():
    return ConformerEncoder(
        feat_in=16,
        n_layers=2,
        d_model=16,
        n_heads=2,
        ff_expansion_factor=2,
        subsampling="striding",
        subsampling_factor=4,
        subsampling_conv_channels=4,
        conv_kernel_size=5,
        self_attention_model="rel_pos",
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        dropout_att=0.0,
    )


def _waveforms(lengths, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(int(sum(lengths)), generator=generator, dtype=torch.float64) * 0.1


def _dynamic_batch(audio, lengths):
    lengths = torch.tensor(lengths, dtype=torch.int64)
    cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
    return audio, lengths, cu_seqlens


def _bounded_features(features: PackedEncoderActivations, capacity: PackedCapacity, rows: int, bound: int):
    """The content of dynamic packed features in a capacity buffer, as the bounded featurizer lays it out."""
    data = torch.zeros((rows, features.data.shape[1]), dtype=features.data.dtype)
    data[: features.data.shape[0]] = features.data
    lengths = torch.zeros(capacity.utterances, dtype=torch.int64)
    lengths[: features.lengths.numel()] = features.lengths
    cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
    return PackedEncoderActivations(data, lengths, cu_seqlens, max_seqlen=bound, bounded=True)


def _bounded_batch(audio, lengths, capacity: PackedCapacity):
    """The same batch in capacity buffers: zero samples after the content, zero-length utterance slots."""
    buffer = torch.zeros(capacity.samples, dtype=audio.dtype)
    buffer[: audio.numel()] = audio
    padded_lengths = torch.zeros(capacity.utterances, dtype=torch.int64)
    padded_lengths[: len(lengths)] = torch.tensor(lengths)
    cu_seqlens = torch.nn.functional.pad(padded_lengths.cumsum(0), (1, 0)).to(torch.int32)
    return buffer, padded_lengths, cu_seqlens


class TestPackedCapacityMode:
    """Bounded buffers reproduce the dynamic packed path at the content rows, through features, encoder and loss."""

    lengths = [9_000, 4_100, 12_480]
    capacity = PackedCapacity(utterances=5, samples=32_000, max_samples=16_000)

    def _run(self, preprocessor, encoder, loss, audio, lengths, cu_seqlens, capacity, targets, target_lengths):
        features = preprocessor.forward_packed(audio, lengths, cu_seqlens, capacity=capacity)
        encoded = encoder.forward_sequence_packed(features)
        log_probs = torch.log_softmax(
            encoded.data @ torch.ones(encoded.data.shape[1], 7, dtype=encoded.data.dtype) * 0.1
            + torch.arange(7, dtype=encoded.data.dtype) * 0.01,
            dim=-1,
        )
        losses = loss.forward_packed(encoded.with_data(log_probs), targets, target_lengths)
        return features, encoded, losses

    def test_bounded_equals_dynamic(self):
        torch.manual_seed(0)
        preprocessor = _preprocessor().double().train()
        encoder = _encoder().double().train()
        loss = CTCLoss(num_classes=6, reduction="none", zero_infinity=True)
        audio = _waveforms(self.lengths)
        targets = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [2, 2, 1, 3]])
        target_lengths = torch.tensor([3, 2, 4])

        dynamic = self._run(
            preprocessor, encoder, loss, *_dynamic_batch(audio, self.lengths), None, targets, target_lengths
        )
        # the running statistics moved in the first pass; reset them so both passes see the same state
        for module in (preprocessor, encoder):
            for name, buffer in module.named_buffers():
                if name.endswith("running_mean"):
                    buffer.zero_()
                elif name.endswith("running_var"):
                    buffer.fill_(1.0)
                elif name.endswith("num_batches_tracked"):
                    buffer.zero_()
        bounded_targets = torch.zeros((self.capacity.utterances, 6), dtype=torch.long)
        bounded_targets[:3, :4] = targets
        bounded_target_lengths = torch.zeros(self.capacity.utterances, dtype=torch.long)
        bounded_target_lengths[:3] = target_lengths
        bounded = self._run(
            preprocessor,
            encoder,
            loss,
            *_bounded_batch(audio, self.lengths, self.capacity),
            self.capacity,
            bounded_targets,
            bounded_target_lengths,
        )

        dyn_features, dyn_encoded, dyn_losses = dynamic
        features, encoded, losses = bounded
        assert features.bounded and encoded.bounded
        frames_capacity, frame_bound = preprocessor.featurizer.packed_frame_capacity(self.capacity)
        assert features.data.shape == (frames_capacity, 16) and features.max_seqlen == frame_bound
        assert features.lengths[:3].tolist() == dyn_features.lengths.tolist() and features.lengths[3:].tolist() == [
            0,
            0,
        ]
        content = int(dyn_features.cu_seqlens[-1])
        # the featurizer runs in float32; the STFT over the longer guard buffer rounds differently
        torch.testing.assert_close(features.data[:content], dyn_features.data, atol=2e-5, rtol=1e-4)
        assert torch.isfinite(features.data).all()

        assert encoded.lengths[:3].tolist() == dyn_encoded.lengths.tolist()
        assert encoded.max_seqlen >= dyn_encoded.max_seqlen
        assert encoded.data.shape[0] >= int(encoded.cu_seqlens[-1])
        encoded_content = int(dyn_encoded.cu_seqlens[-1])
        torch.testing.assert_close(encoded.data[:encoded_content], dyn_encoded.data, atol=1e-3, rtol=1e-3)
        assert torch.isfinite(encoded.data).all()
        torch.testing.assert_close(losses[:3], dyn_losses, atol=1e-3, rtol=1e-3)
        assert torch.isfinite(losses).all()

        # the same features in a capacity buffer: the encoder and the loss agree exactly (float64)
        encoder.eval()
        with torch.no_grad():
            dyn_again = encoder.forward_sequence_packed(dyn_features)
            same = _bounded_features(dyn_features, self.capacity, frames_capacity, frame_bound)
            again = encoder.forward_sequence_packed(same)
        torch.testing.assert_close(again.data[:encoded_content], dyn_again.data, atol=1e-9, rtol=1e-9)
        assert torch.isfinite(again.data).all()

        # the padded view of a bounded batch drops the slack rows
        padded = unpack_encoder_output(encoded)
        assert padded.shape == (self.capacity.utterances, encoded.max_seqlen, 16)
        first = int(encoded.lengths[0])
        torch.testing.assert_close(padded[0, :first], encoded.data[:first])
        assert padded[3:].abs().sum() == 0

    def test_bounded_gradients_equal_dynamic(self):
        torch.manual_seed(1)
        preprocessor = _preprocessor().double().train()
        audio = _waveforms(self.lengths, seed=3)
        encoder_a = _encoder().double().train()
        encoder_b = _encoder().double().train()
        encoder_b.load_state_dict(encoder_a.state_dict())
        loss = CTCLoss(num_classes=6, reduction="sum", zero_infinity=True)
        targets = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [2, 2, 1, 3]])
        target_lengths = torch.tensor([3, 2, 4])

        with torch.no_grad():
            dyn_features = preprocessor.forward_packed(*_dynamic_batch(audio, self.lengths))
            rows, bound = preprocessor.featurizer.packed_frame_capacity(self.capacity)
            features = _bounded_features(dyn_features, self.capacity, rows, bound)
        head = torch.randn(16, 7, dtype=torch.float64) * 0.1

        dyn_encoded = encoder_a.forward_sequence_packed(dyn_features)
        dyn_loss = loss.forward_packed(
            dyn_encoded.with_data(torch.log_softmax(dyn_encoded.data @ head, dim=-1)), targets, target_lengths
        )
        dyn_loss.backward()

        bounded_targets = torch.zeros((self.capacity.utterances, 6), dtype=torch.long)
        bounded_targets[:3, :4] = targets
        bounded_target_lengths = torch.zeros(self.capacity.utterances, dtype=torch.long)
        bounded_target_lengths[:3] = target_lengths
        encoded = encoder_b.forward_sequence_packed(features)
        bounded_loss = loss.forward_packed(
            encoded.with_data(torch.log_softmax(encoded.data @ head, dim=-1)), bounded_targets, bounded_target_lengths
        )
        bounded_loss.backward()

        torch.testing.assert_close(bounded_loss, dyn_loss, atol=1e-7, rtol=1e-7)
        for (name, parameter), (_, reference) in zip(encoder_b.named_parameters(), encoder_a.named_parameters()):
            if reference.grad is None:
                assert parameter.grad is None or parameter.grad.abs().sum() == 0, name
                continue
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-7, rtol=1e-6, msg=name)

    def test_batch_means_count_the_utterances_in_use(self):
        """The CTC batch means of a bounded batch divide by the utterances with frames, not by the slots."""
        torch.manual_seed(4)
        lengths = torch.tensor([6, 4, 0, 0], dtype=torch.int64)
        cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
        log_probs = torch.log_softmax(torch.randn(12, 5, dtype=torch.float64), dim=-1)
        bounded = PackedEncoderActivations(log_probs, lengths, cu_seqlens, max_seqlen=6, bounded=True)
        dynamic = PackedEncoderActivations(log_probs[:10].clone(), lengths[:2], cu_seqlens[:3], max_seqlen=6)
        targets = torch.tensor([[1, 2, 0], [3, 0, 0], [0, 0, 0], [0, 0, 0]])
        target_lengths = torch.tensor([2, 1, 0, 0])
        for reduction in ("mean_batch", "mean", "mean_volume", "sum"):
            loss = CTCLoss(num_classes=4, reduction=reduction, zero_infinity=True)
            expected = loss.forward_packed(dynamic, targets[:2], target_lengths[:2])
            actual = loss.forward_packed(bounded, targets, target_lengths)
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9, msg=reduction)

    def test_slot_layout_and_context_bounds(self):
        lengths = torch.tensor([5, 0, 3, 0], dtype=torch.int64)
        cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
        data = torch.zeros(12, 2)
        packed = PackedEncoderActivations(data, lengths, cu_seqlens, max_seqlen=6, bounded=True)
        layout = packed_slot_layout(packed, alignment=2, gap=3)
        assert layout.total_slots == 12 + 4 * (3 + 1) + 2
        assert layout.token_valid.tolist() == [True] * 8 + [False] * 4
        assert layout.token_slots[8:].tolist() == [layout.total_slots - 1] * 4
        assert layout.token_slots[:5].tolist() == list(range(5))
        # the empty slot still owns its gap, the third utterance starts after it: slots 8 + 4 = 12, 13, 14
        assert layout.slot_offsets.tolist() == [0, 8, 12, 18]
        assert layout.token_slots[5:8].tolist() == [12, 13, 14]
        scattered = layout.scatter(torch.arange(12, dtype=torch.float32).unsqueeze(1).expand(12, 2))
        assert torch.equal(
            layout.gather(scattered)[:8], torch.arange(8, dtype=torch.float32).unsqueeze(1).expand(8, 2)
        )
        context = PackedAttentionContext(packed)
        assert context.bounded and context.total_tokens == 12 and context.max_seqlen == 6
        assert context.valid_rows().tolist() == [True] * 8 + [False] * 4
        from_lengths = PackedAttentionContext.from_lengths(lengths, total_capacity=12, max_seqlen=6)
        assert from_lengths.bounded and from_lengths.cu_seqlens.tolist() == cu_seqlens.tolist()
        with pytest.raises(ValueError):
            PackedAttentionContext.from_lengths(lengths, total_capacity=12)
