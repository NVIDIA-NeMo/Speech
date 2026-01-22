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


import torch
from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.inference.factory.buffered_pipeline_builder import BufferedPipelineBuilder
from nemo.collections.asr.inference.factory.cache_aware_pipeline_builder import CacheAwarePipelineBuilder
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.utils.enums import PipelineType
from nemo.utils import logging


class PipelineBuilder:
    """Router for building the pipeline based on the pipeline type."""

    @staticmethod
    def set_matmul_precision(matmul_precision: str) -> None:
        """
        Set the matmul precision.
        Args:
            matmul_precision: (str) Matmul precision: highest, high, medium
        """
        choices = ["highest", "high", "medium"]
        matmul_precision = matmul_precision.lower()
        if matmul_precision not in choices:
            raise ValueError(f"Invalid matmul precision: {matmul_precision}. Need to be one of {choices}")
        torch.set_float32_matmul_precision(matmul_precision)
        logging.info(f"Using matmul precision: {matmul_precision}")

    @staticmethod
    def set_log_level(log_level: int) -> None:
        """
        Set the logging level.
        Args:
            log_level: (int) Logging level: 0 (NOTSET), 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL)
        """
        choices = [0, 10, 20, 30, 40, 50]
        if log_level not in choices:
            raise ValueError(f"Invalid log level: {log_level}. Need to be one of {choices}")
        logging.setLevel(log_level)

    @staticmethod
    def build_pipeline(cfg: DictConfig) -> BasePipeline:
        """
        Build the pipeline based on the config.
        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns Pipeline object
        """
        PipelineBuilder.set_log_level(cfg.log_level)
        PipelineBuilder.set_matmul_precision(cfg.matmul_precision)
        pipeline_type = PipelineType.from_str(cfg.pipeline_type)
        if pipeline_type is PipelineType.BUFFERED:
            builder = BufferedPipelineBuilder
        elif pipeline_type is PipelineType.CACHE_AWARE:
            builder = CacheAwarePipelineBuilder
        else:
            raise ValueError(f"Invalid pipeline type: {cfg.pipeline_type}")

        return builder.build(cfg)
