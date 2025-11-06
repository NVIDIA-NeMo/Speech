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

from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.inference.factory.base_builder import BaseBuilder
from nemo.collections.asr.inference.pipelines.cache_aware_ctc_pipeline import CacheAwareCTCPipeline
from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import CacheAwareRNNTPipeline
from nemo.collections.asr.inference.utils.enums import ASRDecodingType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.utils import logging


class CacheAwarePipelineBuilder(BaseBuilder):
    """
    Cache Aware Pipeline Builder class.
    Builds the cache aware CTC/RNNT pipelines.
    """

    @classmethod
    def build(cls, cfg: DictConfig) -> CacheAwareCTCPipeline | CacheAwareRNNTPipeline:
        """
        Build the cache aware streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareCTCPipeline or CacheAwareRNNTPipeline object
        """
        asr_decoding_type = ASRDecodingType.from_str(cfg.asr_decoding_type)

        if asr_decoding_type is ASRDecodingType.RNNT:
            return cls.build_cache_aware_rnnt_pipeline(cfg)
        elif asr_decoding_type is ASRDecodingType.CTC:
            return cls.build_cache_aware_ctc_pipeline(cfg)

        raise ValueError("Invalid asr decoding type for cache aware streaming. Need to be one of ['CTC', 'RNNT']")

    @classmethod
    def get_rnnt_decoding_cfg(cls) -> RNNTDecodingConfig:
        """
        Get the decoding config for the RNNT pipeline.
        Returns:
            (RNNTDecodingConfig) Decoding config
        """
        decoding_cfg = RNNTDecodingConfig()
        decoding_cfg.strategy = "greedy_batch"
        decoding_cfg.preserve_alignments = False
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        decoding_cfg.greedy.max_symbols = 10
        decoding_cfg.fused_batch_size = -1
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
        decoding_cfg.preserve_alignments = False
        return decoding_cfg

    @classmethod
    def build_cache_aware_rnnt_pipeline(cls, cfg: DictConfig) -> CacheAwareRNNTPipeline:
        """
        Build the cache aware RNNT streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareRNNTPipeline object
        """
        # building ASR model
        decoding_cfg = cls.get_rnnt_decoding_cfg()
        asr_model = cls._build_asr(cfg, decoding_cfg)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building cache aware RNNT pipeline
        ca_rnnt_pipeline = CacheAwareRNNTPipeline(cfg, asr_model, itn_model=itn_model)
        logging.info(f"`{type(ca_rnnt_pipeline).__name__}` pipeline loaded")
        return ca_rnnt_pipeline

    @classmethod
    def build_cache_aware_ctc_pipeline(cls, cfg: DictConfig) -> CacheAwareCTCPipeline:
        """
        Build the cache aware CTC streaming pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns CacheAwareCTCPipeline object
        """
        # building ASR model
        decoding_cfg = cls.get_ctc_decoding_cfg()
        asr_model = cls._build_asr(cfg, decoding_cfg)

        # building ITN model
        itn_model = cls._build_itn(cfg, input_is_lower_cased=True)

        # building cache aware CTC pipeline
        ca_ctc_pipeline = CacheAwareCTCPipeline(cfg, asr_model, itn_model=itn_model)
        logging.info(f"`{type(ca_ctc_pipeline).__name__}` pipeline loaded")
        return ca_ctc_pipeline
