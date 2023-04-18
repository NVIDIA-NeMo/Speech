# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


class TTSDataType:
    """Represent TTSDataType."""

    name = None


class WithLens:
    """Represent that this data type also returns lengths for data."""


class Audio(TTSDataType, WithLens):
    name = "audio"


class Text(TTSDataType, WithLens):
    name = "text"


class LogMel(TTSDataType, WithLens):
    name = "log_mel"


class Durations(TTSDataType):
    name = "durations"


class AlignPriorMatrix(TTSDataType):
    name = "align_prior_matrix"


class Pitch(TTSDataType, WithLens):
    name = "pitch"


class Energy(TTSDataType, WithLens):
    name = "energy"


class SpeakerID(TTSDataType):
    name = "speaker_id"


class Voiced_mask(TTSDataType):
    name = "voiced_mask"


class P_voiced(TTSDataType):
    name = "p_voiced"


class LMTokens(TTSDataType):
    name = "lm_tokens"


class ReferenceAudio(TTSDataType, WithLens):
    name = "reference_audio"


MAIN_DATA_TYPES = [Audio, Text]
VALID_SUPPLEMENTARY_DATA_TYPES = [
    LogMel,
    Durations,
    AlignPriorMatrix,
    Pitch,
    Energy,
    SpeakerID,
    LMTokens,
    Voiced_mask,
    P_voiced,
    ReferenceAudio,
]
DATA_STR2DATA_CLASS = {d.name: d for d in MAIN_DATA_TYPES + VALID_SUPPLEMENTARY_DATA_TYPES}
