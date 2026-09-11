# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from easymagpie_vllm_omni.codec.encoder import EasyMagpieCodecEncoder
from easymagpie_vllm_omni.codec.encoder_config import EasyMagpieCodecEncoderConfig
from easymagpie_vllm_omni.codec.reference_speaker_encoder import EasyMagpieReferenceSpeakerEncoder
from easymagpie_vllm_omni.codec.weight_conversion import convert_encoder_state_dict


def _tiny_config() -> EasyMagpieCodecEncoderConfig:
    return EasyMagpieCodecEncoderConfig(
        sample_rate=16000,
        samples_per_frame=8,
        output_dim=4,
        resolutions=[[8, 2, 8], [16, 4, 16]],
        resolution_filters=[4, 6],
        downsample_filters=[8],
        downsample_rates=[2],
        original_num_codebooks=1,
        original_num_levels_per_group=[4, 4, 4, 4],
        num_codebooks=2,
        num_levels_per_group=[4, 4],
        frame_stacking_factor=2,
        audio_bos_id=16,
        audio_eos_id=17,
        context_audio_bos_id=18,
        context_audio_eos_id=19,
        embedding_dim=4,
        reference_speaker_encoder_n_layers=1,
        reference_speaker_encoder_d_ffn=8,
        reference_speaker_encoder_n_heads=2,
        reference_speaker_encoder_kernel_size=1,
        reference_speaker_encoder_max_length=16,
    )


@torch.inference_mode()
def test_encoder_matches_speech_encoder_and_fsq_repacking_for_a_batch():
    from nemo.collections.tts.modules.audio_codec_modules import (
        GroupFiniteScalarQuantizer,
        MultiResolutionSTFTEncoder,
        VectorQuantizerIndexConverter,
    )

    torch.manual_seed(7)
    config = _tiny_config()
    speech_encoder = MultiResolutionSTFTEncoder(
        out_dim=config.output_dim,
        resolutions=config.resolutions,
        resolution_filter_list=config.resolution_filters,
        down_sample_filter_list=config.downsample_filters,
        down_sample_rate_list=config.downsample_rates,
        kernel_size=config.kernel_size,
        activation=config.activation,
        pad_mode=config.pad_mode,
    ).eval()
    state = {f"audio_encoder.{name}": value for name, value in speech_encoder.state_dict().items()}

    encoder = EasyMagpieCodecEncoder(config).eval()
    converted = convert_encoder_state_dict(state)
    encoder.audio_encoder.load_state_dict(
        {name.removeprefix("audio_encoder."): value for name, value in converted.items()}, strict=True
    )

    audio = torch.randn(2, 32)
    audio_lens = torch.tensor([32, 24])
    expected_encoded, expected_lens = speech_encoder(audio=audio, audio_len=audio_lens)
    actual_encoded, actual_lens = encoder.audio_encoder(audio, audio_lens)
    torch.testing.assert_close(actual_lens, expected_lens)
    torch.testing.assert_close(actual_encoded, expected_encoded, rtol=1e-5, atol=1e-6)

    original_fsq = GroupFiniteScalarQuantizer(
        num_groups=config.original_num_codebooks,
        num_levels_per_group=config.original_num_levels_per_group,
    )
    target_fsq = GroupFiniteScalarQuantizer(
        num_groups=config.num_codebooks,
        num_levels_per_group=config.num_levels_per_group,
    )
    converter = VectorQuantizerIndexConverter(original_fsq, target_fsq)
    original_codes = original_fsq.encode(inputs=expected_encoded.float(), input_len=expected_lens).permute(1, 0, 2)
    expected_codes = converter.convert_original_to_new(original_codes, expected_lens)

    actual_codes, actual_code_lens = encoder.encode(audio, audio_lens)
    torch.testing.assert_close(actual_code_lens, expected_lens)
    torch.testing.assert_close(actual_codes, expected_codes)


def test_stack_codes_matches_multiturn_time_to_channel_layout():
    encoder = EasyMagpieCodecEncoder(_tiny_config())
    codes = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 0], [9, 10, 0]],
        ],
        dtype=torch.int32,
    )
    code_lens = torch.tensor([3, 2])

    stacked, stacked_lens = encoder.stack_codes(codes, code_lens)

    expected = torch.tensor(
        [
            [[1, 3], [2, 17], [4, 6], [5, 17]],
            [[7, 0], [8, 17], [9, 0], [10, 17]],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(stacked, expected)
    torch.testing.assert_close(stacked_lens, torch.tensor([2, 1]))


def test_stack_context_codes_preserves_bos_and_stacks_the_eos_terminated_body():
    encoder = EasyMagpieCodecEncoder(_tiny_config())
    codes = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.int32)

    stacked, stacked_lens = encoder.stack_context_codes(
        codes,
        torch.tensor([3]),
        bos_id=18,
        eos_id=19,
    )

    expected = torch.tensor(
        [[[18, 1, 3], [18, 2, 19], [18, 4, 6], [18, 5, 19]]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(stacked, expected)
    torch.testing.assert_close(stacked_lens, torch.tensor([3]))


@torch.inference_mode()
def test_codec_tower_returns_acoustic_tokens_and_reference_speaker_embeddings_for_every_item():
    config = _tiny_config()
    encoder = EasyMagpieCodecEncoder(config).eval()

    output = encoder(
        torch.randn(2, 32),
        torch.tensor([32, 24]),
        audio_frame_embedder=lambda codes: codes.float(),
    )

    assert output.acoustic_codes.shape == (2, 4, 2)
    torch.testing.assert_close(output.acoustic_lens, torch.tensor([2, 2]))
    assert output.reference_speaker_embeddings is not None
    assert output.reference_speaker_embeddings.shape == (2, 4, 4)
    torch.testing.assert_close(output.reference_speaker_embedding_lens, torch.tensor([4, 3]))


@torch.inference_mode()
def test_codec_tower_output_preserves_legacy_two_tensor_unpacking():
    encoder = EasyMagpieCodecEncoder(_tiny_config()).eval()

    codes, code_lens = encoder(torch.randn(1, 16), torch.tensor([16]))

    assert codes.shape == (1, 4, 1)
    torch.testing.assert_close(code_lens, torch.tensor([1]))


@torch.inference_mode()
def test_reference_speaker_encoder_matches_nemo_transformer():
    from nemo.collections.tts.modules.transformer_2501 import Transformer

    torch.manual_seed(11)
    native = Transformer(
        n_layers=1,
        d_model=4,
        d_ffn=8,
        sa_n_heads=2,
        kernel_size=1,
        p_dropout=0.0,
        is_causal=False,
        use_learnable_pos_emb=True,
        max_length_causal_mask=16,
    ).eval()
    serving = EasyMagpieReferenceSpeakerEncoder(
        n_layers=1,
        d_model=4,
        d_ffn=8,
        n_heads=2,
        kernel_size=1,
        max_length=16,
    ).eval()
    serving.load_state_dict(native.state_dict(), strict=True)

    inputs = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])
    mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
    expected = native(inputs, mask, cond=None, cond_mask=None)["output"]
    actual = serving(inputs, lengths)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_encoder_pads_each_waveform_length_to_a_complete_codec_frame():
    encoder = EasyMagpieCodecEncoder(_tiny_config())
    audio = torch.randn(2, 17)

    padded, padded_lens = encoder._pad_audio(audio, torch.tensor([17, 9]))

    assert padded.shape == (2, 24)
    torch.testing.assert_close(padded_lens, torch.tensor([24, 16]))
    torch.testing.assert_close(padded[0, :17], audio[0])
    torch.testing.assert_close(padded[1, :9], audio[1, :9])
    torch.testing.assert_close(padded[0, 17:], torch.zeros(7))
    torch.testing.assert_close(padded[1, 9:], torch.zeros(15))
