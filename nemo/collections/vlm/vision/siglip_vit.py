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
# pylint: disable=C0301

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import lightning.pytorch as L
import torch

from nemo.collections.llm.fn.activation import openai_gelu

from nemo.collections.vlm.vision.base import CLIPViTConfig
from nemo.lightning import io, teardown


@dataclass
class SigLIPViT400M_14_384_Config(CLIPViTConfig):
    """Siglip so400m patch14 384 config"""

    vision_model_type: str = "siglip"
    patch_dim: int = 14
    img_h: int = 384
    img_w: int = 384
    num_layers: int = 27
    num_attention_heads: int = 16
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1152
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4304
    gated_linear_unit: bool = False
    activation_func: callable = openai_gelu
    kv_channels: int = 72
    num_query_groups: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'LayerNorm'
    apply_rope_fusion: bool = False
    qk_layernorm: bool = False
    layernorm_epsilon: float = 1e-6


class SigLIPViTModel(L.LightningModule, io.IOMixin, io.ConnectorMixin):
    """SigLIP ViT NeMo Wrapper"""

    def __init__(self, config: Optional[CLIPViTConfig] = None):
        # pylint: disable=C0115,C0116
        super().__init__()
        self.config = config

    def configure_model(self) -> None:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "module"):
            self.module = self.config.configure_model()


@io.model_importer(SigLIPViTModel, "hf")
class SigLIPViTImporter(io.ModelConnector["SigLIPVisionModel", SigLIPViTModel]):
    """HF SigLIP ViT Importer"""

    def init(self) -> SigLIPViTModel:
        # pylint: disable=C0115,C0116
        return SigLIPViTModel(self.config)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        from transformers import AutoModel

        source = AutoModel.from_pretrained(str(self), trust_remote_code=True)
        target = self.init()
        trainer = self.nemo_setup(target)

        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted SigLIPViT model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    @property
    def config(self) -> CLIPViTConfig:
        # pylint: disable=C0115,C0116
        from transformers import AutoConfig

        source = AutoConfig.from_pretrained(str(self), trust_remote_code=True)

        patch_dim = source.vision_config.patch_size
        output = SigLIPViT400M_14_384_Config(
            patch_dim=patch_dim,
            hidden_size=source.vision_config.hidden_size,
            img_h=source.vision_config.image_size // patch_dim * patch_dim,
            img_w=source.vision_config.image_size // patch_dim * patch_dim,
            ffn_hidden_size=source.vision_config.intermediate_size,
            num_attention_heads=source.vision_config.num_attention_heads,
            num_layers=source.vision_config.num_hidden_layers,
            kv_channels=source.vision_config.hidden_size // source.vision_config.num_attention_heads,
            num_query_groups=source.vision_config.num_attention_heads,
        )
        return output

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116
        mapping = {}
        mapping.update(
            {
                "vision_model.embeddings.patch_embedding.weight": "conv1.weight",
                "vision_model.embeddings.patch_embedding.bias": "conv1.bias",
                "vision_model.embeddings.position_embedding.weight": "position_embeddings.weight",
                "vision_model.post_layernorm.weight": "ln_post.weight",
                "vision_model.post_layernorm.bias": "ln_post.bias",
                "vision_model.encoder.layers.*.self_attn.out_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
                "vision_model.encoder.layers.*.self_attn.out_proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
                "vision_model.encoder.layers.*.layer_norm1.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "vision_model.encoder.layers.*.layer_norm1.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
                "vision_model.encoder.layers.*.mlp.fc1.weight": "decoder.layers.*.mlp.linear_fc1.weight",
                "vision_model.encoder.layers.*.mlp.fc1.bias": "decoder.layers.*.mlp.linear_fc1.bias",
                "vision_model.encoder.layers.*.mlp.fc2.weight": "decoder.layers.*.mlp.linear_fc2.weight",
                "vision_model.encoder.layers.*.mlp.fc2.bias": "decoder.layers.*.mlp.linear_fc2.bias",
                "vision_model.encoder.layers.*.layer_norm2.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "vision_model.encoder.layers.*.layer_norm2.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            }
        )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _import_vision_qkv_bias,
                _import_vision_qkv,
            ],
        )


def import_qkv(q, k, v, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
    # pylint: disable=C0115,C0116
    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_vision_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config
    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)


@io.state_transform(
    source_key=(
        "vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vision_qkv(ctx: io.TransformCTX, q, k, v):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config
    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )
