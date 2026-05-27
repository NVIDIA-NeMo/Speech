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

from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.inference.factory.base_builder import BaseBuilder
from nemo.collections.asr.inference.pipelines.buffered_ctc_pipeline import BufferedCTCPipeline
from nemo.collections.asr.inference.pipelines.buffered_rnnt_pipeline import BufferedRNNTPipeline
from nemo.collections.asr.inference.pipelines.buffered_salm_pipeline import BufferedSALMPipeline
from nemo.collections.asr.inference.utils.enums import ASRDecodingType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.utils import logging


class BufferedPipelineBuilder(BaseBuilder):
    """
    Buffered Pipeline Builder class.
    Builds:
        - buffered CTC/RNNT/TDT pipelines.
        - buffered SALM pipeline.
    """

    @classmethod
    def build(cls, cfg: DictConfig) -> BufferedRNNTPipeline | BufferedCTCPipeline | BufferedSALMPipeline:
        """
        Build the buffered streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns BufferedRNNTPipeline, BufferedCTCPipeline, or BufferedSALMPipeline object
        """
        asr_decoding_type = ASRDecodingType.from_str(cfg.asr_decoding_type)

        if asr_decoding_type is ASRDecodingType.RNNT:
            return cls.build_buffered_rnnt_pipeline(cfg)
        elif asr_decoding_type is ASRDecodingType.CTC:
            return cls.build_buffered_ctc_pipeline(cfg)
        elif asr_decoding_type is ASRDecodingType.SALM:
            return cls.build_buffered_salm_pipeline(cfg)

        raise ValueError("Invalid asr decoding type for buffered streaming. Need to be one of ['CTC', 'RNNT', 'SALM']")

    @classmethod
    def get_rnnt_decoding_cfg(cls, cfg: DictConfig) -> RNNTDecodingConfig:
        """
        Get the decoding config for the RNNT pipeline.
        Returns:
            (RNNTDecodingConfig) Decoding config
        """
        base_cfg_structured = OmegaConf.structured(RNNTDecodingConfig)
        base_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg_structured))
        decoding_cfg = OmegaConf.merge(base_cfg, cfg.asr.decoding)
        return decoding_cfg

    @classmethod
    def get_ctc_decoding_cfg(cls) -> CTCDecodingConfig:
        """
        Get the decoding config for the CTC pipeline.
        Returns:
            (CTCDecodingConfig) Decoding config
        """
        decoding_cfg = CTCDecodingConfig()
        decoding_cfg.strategy = "greedy"
        return decoding_cfg

    @classmethod
    def build_buffered_rnnt_pipeline(cls, cfg: DictConfig) -> BufferedRNNTPipeline:
        """
        Build the RNNT streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns BufferedRNNTPipeline object
        """
        # building ASR model
        decoding_cfg = cls.get_rnnt_decoding_cfg(cfg)
        asr_model = cls._build_asr(cfg, decoding_cfg)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building NMT model
        nmt_model = cls._build_nmt(cfg)

        # building RNNT pipeline
        rnnt_pipeline = BufferedRNNTPipeline(cfg, asr_model, itn_model, nmt_model)
        logging.info(f"`{type(rnnt_pipeline).__name__}` pipeline loaded")
        return rnnt_pipeline

    @classmethod
    def build_buffered_ctc_pipeline(cls, cfg: DictConfig) -> BufferedCTCPipeline:
        """
        Build the CTC buffered streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns BufferedCTCPipeline object
        """
        # building ASR model
        decoding_cfg = cls.get_ctc_decoding_cfg()
        asr_model = cls._build_asr(cfg, decoding_cfg)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building NMT model
        nmt_model = cls._build_nmt(cfg)

        # building CTC pipeline
        ctc_pipeline = BufferedCTCPipeline(cfg, asr_model, itn_model, nmt_model)
        logging.info(f"`{type(ctc_pipeline).__name__}` pipeline loaded")
        return ctc_pipeline

    @classmethod
    def build_buffered_salm_pipeline(cls, cfg: DictConfig) -> BufferedSALMPipeline:
        """
        Build the buffered SALM pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns BufferedSALMPipeline object
        """
        # building ASR model
        asr_model = cls._build_asr(cfg, decoding_cfg=None)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building NMT model
        nmt_model = cls._build_nmt(cfg)

        # building SALM pipeline
        salm_pipeline = BufferedSALMPipeline(cfg, asr_model, itn_model, nmt_model)
        logging.info(f"`{type(salm_pipeline).__name__}` pipeline loaded")
        return salm_pipeline
