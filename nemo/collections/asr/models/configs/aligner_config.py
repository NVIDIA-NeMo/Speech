# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.k2.classes import GraphModuleConfig


@dataclass
class AlignerCTCConfig:
    prob_suppress_index: int = -1
    prob_suppress_value: float = 1.0


@dataclass
class AlignerRNNTConfig:
    predictor_window_size: int = 0
    predictor_step_size: int = 1


@dataclass
class AlignerWrapperModelConfig:
    alignment_type: str = "forced"
    word_output: bool = True
    cpu_decoding: bool = False
    decode_batch_size: int = 0
    ctc_cfg: AlignerCTCConfig = AlignerCTCConfig()
    rnnt_cfg: AlignerRNNTConfig = AlignerRNNTConfig()


@dataclass
class K2AlignerWrapperModelConfig(AlignerWrapperModelConfig):
    decoder_module_cfg: GraphModuleConfig = GraphModuleConfig()
