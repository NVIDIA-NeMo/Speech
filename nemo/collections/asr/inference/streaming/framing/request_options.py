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
from typing import Any, TypeAlias

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
    language_code: str | None = None
    enable_nmt: bool = None
    source_language: str = None
    target_language: str = None

    def __post_init__(self) -> None:
        """
        Post-init hook:
            Converts the asr_output_granularity to ASROutputGranularity if it is a string
        """
        if isinstance(self.asr_output_granularity, str):
            self.asr_output_granularity = ASROutputGranularity.from_str(self.asr_output_granularity)

        if not self.enable_nmt:
            # Forcibly set the source and target languages to None
            self.source_language = None
            self.target_language = None

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

    @staticmethod
    def _with_default(value: Any, default: Any) -> Any:
        """
        Return the value if it is not None, otherwise return the default value.
        Args:
            value: The value to check.
            default: The default value to return if the value is None.
        Returns:
            The value if it is not None, otherwise return the default value.
        """
        return default if value is None else value

    def augment_with_defaults(
        self,
        default_enable_itn: bool,
        default_enable_pnc: bool,
        default_enable_nmt: bool,
        default_source_language: str,
        default_target_language: str,
        default_stop_history_eou: int,
        default_asr_output_granularity: ASROutputGranularity | str,
        default_language_code: str | None = None,
    ) -> "ASRRequestOptions":
        """
        Fill unset fields with the passed default values.
        Args:
            default_enable_itn (bool): Default enable ITN.
            default_enable_pnc (bool): Default enable PNC.
            default_enable_nmt (bool): Default enable NMT.
            default_source_language (str): Default source language.
            default_target_language (str): Default target language.
            default_stop_history_eou (int): Default stop history EOU.
            default_asr_output_granularity (ASROutputGranularity | str): Default output granularity.
            default_language_code (str | None): Default language code for prompt-enabled models.
        Returns:
            ASRRequestOptions: Augmented options.
        """
        if isinstance(default_asr_output_granularity, str):
            default_asr_output_granularity = ASROutputGranularity.from_str(default_asr_output_granularity)

        enable_itn = self._with_default(self.enable_itn, default_enable_itn)
        enable_pnc = self._with_default(self.enable_pnc, default_enable_pnc)
        enable_nmt = self._with_default(self.enable_nmt, default_enable_nmt)
        if not enable_nmt:
            # Forcibly set the source and target languages to None
            source_language, target_language = None, None
        else:
            source_language = self._with_default(self.source_language, default_source_language)
            target_language = self._with_default(self.target_language, default_target_language)

        stop_history_eou = self._with_default(self.stop_history_eou, default_stop_history_eou)
        granularity = self._with_default(self.asr_output_granularity, default_asr_output_granularity)
        language_code = self._with_default(self.language_code, default_language_code)

        return ASRRequestOptions(
            enable_itn=enable_itn,
            enable_pnc=enable_pnc,
            enable_nmt=enable_nmt,
            source_language=source_language,
            target_language=target_language,
            stop_history_eou=stop_history_eou,
            asr_output_granularity=granularity,
            language_code=language_code,
        )


RequestOptions: TypeAlias = ASRRequestOptions
