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

from functools import partial
from typing import Literal, Optional

from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.diffusion.models.dit import dit_embeddings
from nemo.collections.diffusion.models.dit.dit_model import DiTCrossAttentionModel
from nemo.collections.diffusion.models.dit_llama.dit_llama_layer_spec import get_dit_llama_spec


class DiTLlamaModel(DiTCrossAttentionModel):
    """DiT-LLaMA model."""

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        max_img_h: int = 80,
        max_img_w: int = 80,
        max_frames: int = 34,
        patch_spatial: int = 1,
        patch_temporal: int = 1,
        in_channels: int = 16,
        out_channels: int = 16,
        vp_stage: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            position_embedding_type=position_embedding_type,
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            max_frames=max_frames,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            in_channels=in_channels,
            out_channels=out_channels,
            transformer_decoder_layer_spec=partial(
                get_dit_llama_spec, num_experts=config.num_moe_experts, attn_mask_type=config.attn_mask_type
            ),
            pos_embedder=dit_embeddings.FactorizedLearnable3DEmbedding,
            vp_stage=vp_stage,
            **kwargs,
        )
