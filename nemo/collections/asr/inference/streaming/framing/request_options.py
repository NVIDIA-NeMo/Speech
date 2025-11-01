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


from dataclasses import dataclass
from typing import TypeAlias
from nemo.collections.asr.inference.utils.enums import ASROutputGranularity


@dataclass(slots=True)
class ASRRequestOptions:
    """
    Immutable dataclass representing options for a request
    None value means that the option is not set and the default value will be used
    """

    enable_itn: bool = None
    enable_pnc: bool = None
    stop_history_eou: int = None
    asr_output_granularity: ASROutputGranularity | str = None

    def __post_init__(self) -> None:
        """
        Post-init hook:
            Converts the asr_output_granularity to ASROutputGranularity if it is a string
        """
        if isinstance(self.asr_output_granularity, str):
            self.asr_output_granularity = ASROutputGranularity.from_str(self.asr_output_granularity)

    def is_word_level_output(self) -> bool:
        """
        Check if the output granularity is word level.
        """
        return self.asr_output_granularity is ASROutputGranularity.WORD

    def is_segment_level_output(self) -> bool:
        """
        Check if the output granularity is segment level.
        """
        return self.asr_output_granularity is ASROutputGranularity.SEGMENT

    def augment_with_defaults(
        self,
        default_enable_itn: bool,
        default_enable_pnc: bool,
        default_stop_history_eou: int,
        default_asr_output_granularity: ASROutputGranularity | str,
    ) -> "ASRRequestOptions":
        """
        Augment the options with the default values.
        Args:
            default_enable_itn (bool): Default enable ITN.
            default_enable_pnc (bool): Default enable PNC.
            default_stop_history_eou (int): Default stop history EOU.
            default_asr_output_granularity (ASROutputGranularity | str): Default output granularity.
        Returns:
            ASRRequestOptions: Augmented options.
        """
        if isinstance(default_asr_output_granularity, str):
            default_asr_output_granularity = ASROutputGranularity.from_str(default_asr_output_granularity)
        return ASRRequestOptions(
            enable_itn=default_enable_itn if self.enable_itn is None else self.enable_itn,
            enable_pnc=default_enable_pnc if self.enable_pnc is None else self.enable_pnc,
            stop_history_eou=default_stop_history_eou if self.stop_history_eou is None else self.stop_history_eou,
            asr_output_granularity=(
                default_asr_output_granularity if self.asr_output_granularity is None else self.asr_output_granularity
            ),
        )


RequestOptions: TypeAlias = ASRRequestOptions
