# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from types import SimpleNamespace

import convert_codec as converter
import pytest
from easymagpie_vllm_omni.codec.encoder_config import EasyMagpieCodecEncoderConfig


def _valid_decoder_config() -> dict:
    return {
        "_target_": "nemo.collections.tts.modules.audio_codec_modules.ResNetDecoder",
        "is_causal": True,
        "activation": "half_snake",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("_target_", "some.OtherDecoder", "ResNetDecoder"),
        ("is_causal", False, "causal"),
        ("activation", "snake", "half_snake"),
    ],
)
def test_validate_decoder_config_rejects_unsupported_codec(field, value, message):
    config = _valid_decoder_config()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        converter.validate_decoder_config(config)


def _valid_encoder_config() -> dict:
    return {
        "_target_": "nemo.collections.tts.modules.audio_codec_modules.MultiResolutionSTFTEncoder",
        "out_dim": 4,
        "resolutions": [[8, 2, 8], [16, 4, 16]],
        "resolution_filter_list": [4, 6],
        "down_sample_filter_list": [8],
        "down_sample_rate_list": [2],
        "activation": "lrelu",
        "pad_mode": "replicate",
    }


def _valid_quantizer_config() -> dict:
    return {
        "_target_": "nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer",
        "num_groups": 1,
        "num_levels_per_group": [4, 4, 4, 4],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("_target_", "some.OtherEncoder", "MultiResolutionSTFTEncoder"),
        ("activation", "snake", "lrelu"),
        ("pad_mode", "reflect", "replicate"),
    ],
)
def test_validate_encoder_config_rejects_unsupported_encoder(field, value, message):
    config = _valid_encoder_config()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        converter.validate_encoder_config(config, _valid_quantizer_config())


def test_validate_encoder_config_rejects_unsupported_quantizer():
    quantizer = _valid_quantizer_config()
    quantizer["_target_"] = "some.OtherQuantizer"

    with pytest.raises(ValueError, match="GroupFiniteScalarQuantizer"):
        converter.validate_encoder_config(_valid_encoder_config(), quantizer)


def test_build_encoder_config_captures_codec_and_target_fsq_layouts():
    nemo_config = {
        "sample_rate": 16000,
        "samples_per_frame": 8,
        "audio_encoder": _valid_encoder_config(),
        "vector_quantizer": _valid_quantizer_config(),
    }
    args = SimpleNamespace(
        num_codebooks=2,
        num_levels_per_group=[4, 4],
        frame_stacking_factor=2,
    )

    config = converter.build_encoder_config(nemo_config, args)

    assert isinstance(config, EasyMagpieCodecEncoderConfig)
    assert config.original_num_codebooks == 1
    assert config.original_num_levels_per_group == [4, 4, 4, 4]
    assert config.num_codebooks == 2
    assert config.num_levels_per_group == [4, 4]
    assert config.num_stacked_codebooks == 4
