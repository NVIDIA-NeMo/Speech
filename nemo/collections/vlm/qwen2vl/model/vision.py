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

from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


class VisionRotaryEmbedding(torch.nn.Module):
    # pylint: disable=C0115,C0116

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2VisionModel(VisionModule):
    """Qwen2-VL vision model."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        add_class_token: bool = False,
        class_token_len: int = 1,
        patch_dim: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        spatial_patch_size: int = 14,
        img_h: int = 336,
        img_w: int = 336,
    ) -> None:

        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.embed_dim
        self.patch_dim = patch_dim
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.merge_hidden_size = self.visual_hidden_size * (spatial_merge_size**2)
        self.img_h = img_h
        self.img_w = img_w
        self.in_channels = 3

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len

        self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

        kernel_size = [temporal_patch_size, patch_dim, patch_dim]
        self.conv1 = torch.nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.visual_hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

        head_dim = transformer_config.embed_dim // transformer_config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.add_class_token = add_class_token
        if self.add_class_token:
            self.class_token = torch.nn.Parameter(torch.randn(1, self.class_token_len, self.visual_hidden_size))

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers.
        # TODO: Make pre_process and post_process configurable.
        # NOTE: a final layer norm and/or linear layer in some implementations are omitted here.
        # They can be added separately where needed.
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def rot_pos_emb(self, grid_thw):
        # pylint: disable=C0115,C0116
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_packed_seq_params(self, grid_thw):
        # pylint: disable=C0115,C0116
        from megatron.core.packed_seq_params import PackedSeqParams

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        cu_seqlens = cu_seqlens.squeeze()  # remove batch size dimension (mbs=1)
        # remove -1 "paddings" added in collate_fn
        # cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

        # pre-compute max_seqlens in dataset class for perf
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        # these args are passed eventually into TEDotProductAttention.forward()
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format='thd',
        )

    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of the Qwen2 Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w]
            grid_thw (torch.Tensor): The temporal, height and width of feature shape of each image/frame.
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.

        Returns:
            x (torch.Tensor): output after final transformer block.
        """
        # pylint: disable=C0301
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_dim, self.patch_dim)
        x = self.conv1(x).view(-1, self.visual_hidden_size)  # [seqlen, hidden_size]
        # add batch dim
        x = x.unsqueeze(1)  # [seqlen, 1, hidden_size], THD format, bs=1
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        # from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L158
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L164
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]

        packed_seq_params = self.get_packed_seq_params(grid_thw)
        x = self.decoder(x, attention_mask, rotary_pos_emb=rotary_pos_emb, packed_seq_params=packed_seq_params)

        x = x.squeeze(1).view(-1, self.merge_hidden_size)
        return x
