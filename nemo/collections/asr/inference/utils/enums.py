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


from enum import Enum, auto


class StrEnumMixin:
    @classmethod
    def from_str(cls, name: str):
        """Convert a string to an Enum value (case-insensitive)."""
        normalized = name.lower()
        for member in cls:
            if member.name.lower() == normalized or str(member.value).lower() == normalized:
                return member

        choices = [member.name.lower() for member in cls]
        raise ValueError(f"Invalid {cls.__name__} `{name}`: must be one of {choices}")


class ASRDecodingType(StrEnumMixin, Enum):
    """
    Enumeration of the ASR decoding types.
    """

    CTC = auto()
    RNNT = auto()


class ASROutputGranularity(StrEnumMixin, Enum):
    """
    Enumeration of the ASR output granularity.
    """

    WORD = auto()
    SEGMENT = auto()


class PipelineType(StrEnumMixin, Enum):
    """
    Enumeration of the pipeline types.
    """

    BUFFERED = auto()
    CACHE_AWARE = auto()


class RequestType(StrEnumMixin, Enum):
    """
    Enumeration of the request types.
    """

    FRAME = auto()
    FEATURE_BUFFER = auto()


class FeatureBufferPaddingMode(StrEnumMixin, Enum):
    """
    Enumeration of the feature buffer padding modes.
    """

    LEFT = auto()
    RIGHT = auto()
