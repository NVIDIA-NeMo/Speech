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

import convert_codec as converter
import pytest


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
