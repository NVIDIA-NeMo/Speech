# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.lightning.io.mixin import track_io


@dataclass
class TPOverlapCfg:
    """Dataclass for linear layer TP overlap config."""

    pass


@dataclass
class PipelineOverlapCfg(TPOverlapCfg):
    """Dataclass for pipeline TP overlap config."""

    num_sm: int
    cga_size: int
    num_splits: int
    set_sm_margin: bool
    fp8_buf: bool = (False,)
    atomic_gemm: bool = False
    method: str = "pipeline"


@dataclass
class RingExchangeOverlapCfg(TPOverlapCfg):
    """Dataclass for ring exchange TP overlap config."""

    aggregate: bool = False
    method: str = "ring_exchange"
    num_sm: int = 1
    cga_size: int = 1
    set_sm_margin: bool = False
    fp8_buf: bool = False
    atomic_gemm: bool = False


@dataclass
class BulkOverlapCfg(TPOverlapCfg):
    """Dataclass for bulk TP overlap config."""

    num_sm: int
    cga_size: int
    set_sm_margin: bool
    method: str = "bulk"


@dataclass
class TransformerLayerTPOverlapCfg:
    """Dataclass for transformer layer TP overlap config."""

    qkv_dgrad: TPOverlapCfg
    qkv_wgrad: TPOverlapCfg
    fc1_dgrad: TPOverlapCfg
    fc1_wgrad: TPOverlapCfg
    qkv_fprop: TPOverlapCfg
    proj_dgrad: TPOverlapCfg
    fc1_fprop: TPOverlapCfg
    fc2_dgrad: TPOverlapCfg
    proj_fprop: TPOverlapCfg
    fc2_fprop: TPOverlapCfg


track_io(TPOverlapCfg)
track_io(TransformerLayerTPOverlapCfg)

# TODO: Add more configs and create a getter function for expose a single api
# Model configs: H100/70B/TP8/MBS1/SeqLen8K
userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=32, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

# llama3.1 405b
userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=True),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=True),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=True),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=True),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=8, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=True),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=True),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=True),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=True),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=32, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=8, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=32, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

# llama3 70b LoRA
userbuffers_fp8_h100_h8192_tp2_mbs1_seqlen4096_lora = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=None,
    fc1_dgrad=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc1_wgrad=None,
    qkv_fprop=RingExchangeOverlapCfg(set_sm_margin=True),
    proj_dgrad=RingExchangeOverlapCfg(set_sm_margin=True),
    fc1_fprop=RingExchangeOverlapCfg(set_sm_margin=True),
    fc2_dgrad=RingExchangeOverlapCfg(set_sm_margin=True),
    proj_fprop=RingExchangeOverlapCfg(cga_size=2, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=RingExchangeOverlapCfg(cga_size=2, set_sm_margin=True, fp8_buf=True),
)

# llama3.1 405b LoRA
userbuffers_fp8_h100_h16384_tp4_mbs1_seqlen2048_lora = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=None,
    fc1_dgrad=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc1_wgrad=None,
    qkv_fprop=RingExchangeOverlapCfg(aggregate=True),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=True),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=True),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=True),
    proj_fprop=PipelineOverlapCfg(num_sm=32, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
)

# GPT3 20b
userbuffers_bf16_h100_h6144_tp2_mbs2_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h6144_tp2_mbs2_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
)

# GPT3 175b
userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_fp8_h100_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_bf16_b200_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_fp8_b200_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

# Nemotron 15B
userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=32, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

# Nemotron 340B
userbuffers_bf16_b200_h18432_tp8_mbs1_seqlen4096 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=32, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_b200_h18432_tp8_mbs1_seqlen4096 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=32, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)
