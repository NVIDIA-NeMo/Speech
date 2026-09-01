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

from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor, SpectrogramAugmentation
from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import unpack_encoder_output
from nemo.collections.speechlm2.modules.perception import AudioPerceptionModule, IdentityConnector
from tests.collections.asr.test_parallel_expert_encoder_ggemm import build_toy_pe_encoder


class _FeaturePassthrough(torch.nn.Module):
    def forward(self, input_signal, length):
        return input_signal, length


def _make_perception() -> AudioPerceptionModule:
    encoder = TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        self_attention_model='rope',
        sync_max_audio_length=False,
    ).eval()
    perception = AudioPerceptionModule.__new__(AudioPerceptionModule)
    torch.nn.Module.__init__(perception)
    perception.preprocessor = _FeaturePassthrough()
    perception._modules['encoder'] = encoder
    perception.modality_adapter = IdentityConnector()
    perception.proj = torch.nn.Linear(32, 24)
    perception.spec_augmentation = None
    perception.rote = None
    return perception.eval()


def test_perception_sequence_packed_matches_legacy_and_preserves_state_dict():
    torch.manual_seed(0)
    perception = _make_perception()
    features = torch.randn(3, 8, 12)
    lengths = torch.tensor([12, 7, 4])
    state_keys = set(perception.state_dict())

    with torch.no_grad():
        legacy, output_lengths = perception(input_signal=features, input_signal_length=lengths)
        legacy_with_encoder = perception(
            input_signal=features,
            input_signal_length=lengths,
            return_encoder_emb=True,
        )
        packed = perception.forward_sequence_packed(input_signal=features, input_signal_length=lengths)

    assert len(legacy_with_encoder) == 3
    torch.testing.assert_close(legacy_with_encoder[0], legacy)
    assert torch.equal(legacy_with_encoder[1], output_lengths)
    restored = unpack_encoder_output(packed, total_length=legacy.shape[1])
    valid = torch.arange(legacy.shape[1])[None, :] < output_lengths[:, None]
    torch.testing.assert_close(restored[valid], legacy[valid], rtol=1e-5, atol=1e-6)
    assert perception.supports_sequence_packed_output
    assert set(perception.state_dict()) == state_keys


def test_perception_sequence_packed_rejects_adapter_that_cannot_preserve_thd():
    perception = _make_perception()
    perception.modality_adapter = torch.nn.Identity()

    assert not perception.supports_sequence_packed_output
    with pytest.raises(ValueError, match="IdentityConnector"):
        perception.forward_sequence_packed(
            input_signal=torch.randn(1, 8, 8),
            input_signal_length=torch.tensor([8]),
        )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("encoder_kind", ["transformer", "moe", "pee"])
def test_perception_packed_waveform_matches_padded_waveform_for_all_new_encoders(encoder_kind, device):
    torch.manual_seed(17)
    perception = _make_waveform_perception(encoder_kind).to(device)
    lengths = torch.tensor([4096, 2600, 1200], dtype=torch.long, device=device)
    audios = torch.randn(3, int(lengths.max()), device=device)
    for row, length in zip(audios, lengths):
        row[int(length) :] = 0.0
    packed_audio_samples = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
    audio_cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    checkpoint = perception.state_dict()

    with torch.no_grad():
        expected, expected_lengths = perception(input_signal=audios, input_signal_length=lengths)
        actual = perception.forward_sequence_packed(
            input_signal=packed_audio_samples,
            input_signal_length=lengths,
            input_signal_cu_seqlens=audio_cu_seqlens,
        )

    restored = unpack_encoder_output(actual, total_length=expected.shape[1])
    valid = torch.arange(expected.shape[1], device=device)[None, :] < expected_lengths[:, None]
    assert torch.equal(actual.lengths, expected_lengths)
    torch.testing.assert_close(restored[valid], expected[valid], rtol=2e-5, atol=3e-6)
    assert set(perception.state_dict()) == set(checkpoint)

    reloaded = _make_waveform_perception(encoder_kind).to(device)
    reloaded.load_state_dict(checkpoint, strict=True)
    with torch.no_grad():
        reloaded_output = reloaded.forward_sequence_packed(
            input_signal=packed_audio_samples,
            input_signal_length=lengths,
            input_signal_cu_seqlens=audio_cu_seqlens,
        )
    torch.testing.assert_close(reloaded_output.data, actual.data, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("encoder_kind", ["transformer", "moe", "pee"])
def test_perception_legacy_forward_accepts_packed_waveform(encoder_kind):
    torch.manual_seed(23)
    perception = _make_waveform_perception(encoder_kind)
    lengths = torch.tensor([4096, 2600, 1200])
    audios = torch.randn(3, int(lengths.max()))
    audios.masked_fill_(torch.arange(audios.shape[1])[None, :] >= lengths[:, None], 0.0)
    packed_audio = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
    cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])

    with torch.no_grad():
        expected, expected_lengths = perception(input_signal=audios, input_signal_length=lengths)
        actual, actual_lengths = perception(
            input_signal=packed_audio,
            input_signal_length=lengths,
            input_signal_cu_seqlens=cu_seqlens,
        )

    assert torch.equal(actual_lengths, expected_lengths)
    valid = torch.arange(expected.shape[1])[None, :] < expected_lengths[:, None]
    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("encoder_kind", ["transformer", "moe", "pee"])
def test_perception_packed_waveform_all_empty_batch(encoder_kind):
    perception = _make_waveform_perception(encoder_kind)
    lengths = torch.tensor([0, 0])

    with torch.no_grad():
        output = perception.forward_sequence_packed(
            input_signal=torch.empty(0),
            input_signal_length=lengths,
            input_signal_cu_seqlens=torch.tensor([0, 0, 0]),
        )

    assert output.data.shape == (0, 24)
    assert output.lengths.tolist() == [0, 0]
    assert output.cu_seqlens.tolist() == [0, 0, 0]


def test_perception_packed_waveform_training_dispatches_packed_spec_augment(monkeypatch):
    perception = _make_waveform_perception("transformer").train()
    perception.spec_augmentation = SpectrogramAugmentation(freq_masks=1, freq_width=2)
    calls = 0
    original = perception.spec_augmentation.forward_packed

    def count_packed(input_spec):
        nonlocal calls
        calls += 1
        return original(input_spec)

    def reject_dense(*args, **kwargs):
        raise AssertionError("packed waveform training must not densify for SpecAugment")

    monkeypatch.setattr(perception.spec_augmentation, "forward_packed", count_packed)
    monkeypatch.setattr(perception.spec_augmentation, "forward", reject_dense)
    lengths = torch.tensor([4096, 2600])
    packed_audio = torch.randn(int(lengths.sum()))
    cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])

    output = perception.forward_sequence_packed(
        input_signal=packed_audio,
        input_signal_length=lengths,
        input_signal_cu_seqlens=cu_seqlens,
    )

    assert calls == 1
    assert output.total_tokens == int(output.lengths.sum())


def _make_waveform_perception(encoder_kind: str) -> AudioPerceptionModule:
    if encoder_kind == "transformer":
        encoder = TransformerEncoder(
            feat_in=8,
            d_model=32,
            n_heads=2,
            n_layers=2,
            subsampling_factor=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="rope",
            sync_max_audio_length=False,
        )
        features = 8
    elif encoder_kind == "moe":
        encoder = MoETransformerEncoder(
            feat_in=8,
            d_model=32,
            n_heads=2,
            n_layers=2,
            subsampling_factor=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="rope",
            moe_num_experts=4,
            moe_top_k=2,
            sync_max_audio_length=False,
        )
        features = 8
    else:
        encoder = build_toy_pe_encoder(always_run_diarization=True)
        features = 128

    perception = AudioPerceptionModule.__new__(AudioPerceptionModule)
    torch.nn.Module.__init__(perception)
    perception.preprocessor = AudioToMelSpectrogramPreprocessor(
        features=features,
        normalize="per_feature",
        dither=0,
        pad_to=0,
    )
    perception._modules["encoder"] = encoder
    perception.modality_adapter = IdentityConnector()
    perception.proj = torch.nn.Linear(encoder.d_model, 24)
    perception.spec_augmentation = None
    perception.rote = None
    return perception.eval()
