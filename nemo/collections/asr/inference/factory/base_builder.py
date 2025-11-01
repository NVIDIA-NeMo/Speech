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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.inference.model_wrappers.asr_inference_wrapper import ASRInferenceWrapper
from nemo.collections.asr.inference.model_wrappers.cache_aware_ctc_inference_wrapper import (
    CacheAwareCTCInferenceWrapper,
)
from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
    CacheAwareRNNTInferenceWrapper,
)
from nemo.collections.asr.inference.model_wrappers.ctc_inference_wrapper import CTCInferenceWrapper
from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper
from nemo.collections.asr.inference.utils.enums import ASRDecodingType, PipelineType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer


class BaseBuilder:
    """
    Base Builder class.
    Builds the ASR/ITN components.
    Derived classes should implement the `build` method which should include the logic of creating concrete pipeline.
    """

    @classmethod
    def _build_asr(cls, cfg: DictConfig, decoding_cfg: CTCDecodingConfig | RNNTDecodingConfig) -> ASRInferenceWrapper:
        """
        Build the ASR model based on the config.
        Args:
            cfg: (DictConfig) Config
            decoding_cfg: (CTCDecodingConfig | RNNTDecodingConfig) Decoding config
        Returns:
            (ASRInferenceWrapper) ASR inference model
        """

        asr_decoding_type = ASRDecodingType.from_str(cfg.asr_decoding_type)
        pipeline_type = PipelineType.from_str(cfg.pipeline_type)
        match (asr_decoding_type, pipeline_type):
            case (ASRDecodingType.CTC, PipelineType.BUFFERED):
                asr_class = CTCInferenceWrapper
            case (ASRDecodingType.RNNT, PipelineType.BUFFERED):
                asr_class = RNNTInferenceWrapper
            case (ASRDecodingType.CTC, PipelineType.CACHE_AWARE):
                asr_class = CacheAwareCTCInferenceWrapper
            case (ASRDecodingType.RNNT, PipelineType.CACHE_AWARE):
                asr_class = CacheAwareRNNTInferenceWrapper
            case _:
                raise ValueError(
                    f"Wrong combination of ASR decoding type and pipeline type: {asr_decoding_type, pipeline_type}"
                )

        asr_model = asr_class(
            model_name=cfg.asr.model_name,
            decoding_cfg=decoding_cfg,
            device=cfg.asr.device,
            device_id=cfg.asr.device_id,
            compute_dtype=cfg.asr.compute_dtype,
            use_amp=cfg.asr.use_amp,
        )

        logging.info(f"ASR model `{cfg.asr.model_name}` loaded")
        return asr_model

    @classmethod
    def _build_itn(cls, cfg: DictConfig, input_is_lower_cased: bool) -> AlignmentPreservingInverseNormalizer | None:
        """
        Build the ITN model based on the config.
        Args:
            cfg: (DictConfig) Config
            input_is_lower_cased: (bool) Whether the input is lower cased
        Returns:
            (AlignmentPreservingInverseNormalizer | None) ITN model
        """
        itn_model = None
        if cfg.enable_itn:
            # Do not remove this import. It is used to avoid nemo_text_processing import when verbatim transcripts is enabled.
            from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer

            input_case = (
                AlignmentPreservingInverseNormalizer.LOWER_CASED
                if input_is_lower_cased
                else AlignmentPreservingInverseNormalizer.UPPER_CASED
            )

            target_lang = getattr(cfg, "lang", getattr(cfg, "target_lang", None))
            if target_lang is None:
                raise ValueError("Language is not specified. Cannot load PnC model.")

            itn_cfg = cfg.itn
            with open_dict(itn_cfg):
                itn_cfg.lang = target_lang
                itn_cfg.input_case = input_case
                itn_cfg.cache_dir = cfg.cache_dir

            itn_model = AlignmentPreservingInverseNormalizer(
                lang=itn_cfg.lang,
                input_case=itn_cfg.input_case,
                whitelist=itn_cfg.whitelist,
                cache_dir=itn_cfg.cache_dir,
                overwrite_cache=itn_cfg.overwrite_cache,
                max_number_of_permutations_per_split=itn_cfg.max_number_of_permutations_per_split,
            )
            logging.info(f"Built inverse text normalizer with the input case: `{input_case}`.")

        if itn_model is not None:
            logging.info("ITN model loaded")
        return itn_model

    @classmethod
    def build(cls, cfg: DictConfig) -> Any:
        """
        Build the pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns object responsible for the inference
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
