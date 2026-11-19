# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest

from nemo.collections.asr.inference.utils.enums import ASRDecodingType, ASROutputGranularity, PaddingMode, PipelineType


class TestEnums:

    @pytest.mark.unit
    def test_ASRDecodingType(self):
        assert ASRDecodingType.from_str("ctc") == ASRDecodingType.CTC
        assert ASRDecodingType.from_str("RNNT") == ASRDecodingType.RNNT
        with pytest.raises(ValueError):
            ASRDecodingType.from_str("invalid")

    @pytest.mark.unit
    def test_ASROutputGranularity(self):
        assert ASROutputGranularity.from_str("word") == ASROutputGranularity.WORD
        assert ASROutputGranularity.from_str("segment") == ASROutputGranularity.SEGMENT
        with pytest.raises(ValueError):
            ASROutputGranularity.from_str("invalid")

    @pytest.mark.unit
    def test_PipelineType(self):
        assert PipelineType.from_str("buffered") == PipelineType.BUFFERED
        assert PipelineType.from_str("cache_aware") == PipelineType.CACHE_AWARE
        with pytest.raises(ValueError):
            PipelineType.from_str("invalid")

    @pytest.mark.unit
    def test_PaddingMode(self):
        assert PaddingMode.from_str("left") == PaddingMode.LEFT
        assert PaddingMode.from_str("right") == PaddingMode.RIGHT
        with pytest.raises(ValueError):
            PaddingMode.from_str("invalid")
