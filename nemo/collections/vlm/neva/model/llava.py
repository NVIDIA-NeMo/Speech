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
# pylint: disable=line-too-long

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional, Union

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.llm import Llama2Config7B, Llama2Config13B, LlamaConfig
from nemo.collections.llm.utils import Config
from nemo.collections.vlm.neva.model.base import NevaConfig, NevaModel
from nemo.collections.vlm.vision.base import HFCLIPVisionConfig, MultimodalProjectorConfig
from nemo.lightning import OptimizerModule, io, teardown

if TYPE_CHECKING:

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


# Note: these Llava configs are copied from the corresponding HF model. You may need to modify the parameter for
# your own needs


@dataclass
class LlavaConfig(NevaConfig):
    """Llava Model Base Config"""

    drop_vision_class_token: bool = True
    freeze_vision_model: bool = True


@dataclass
class Llava15Config7B(LlavaConfig):
    """Llava v1.5 Config 7B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama2Config7B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: HFCLIPVisionConfig(pretrained_model_name_or_path="openai/clip-vit-large-patch14-336")
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=1024, hidden_size=4096, ffn_hidden_size=4096)
    )


@dataclass
class Llava15Config13B(LlavaConfig):
    """Llava v1.5 Config 13B"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama2Config13B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: HFCLIPVisionConfig(pretrained_model_name_or_path="openai/clip-vit-large-patch14-336")
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(input_size=1024, hidden_size=5120, ffn_hidden_size=5120)
    )


class LlavaModel(NevaModel):
    """Llava Model NeMo Wrapper"""

    def __init__(
        self,
        config: Annotated[Optional[LlavaConfig], Config[LlavaConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or LlavaConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(LlavaModel, "hf")
class HFLlavaImporter(io.ModelConnector["LlavaForConditionalGeneration", LlavaModel]):
    """Llava Model HF Importer"""

    def init(self) -> LlavaModel:
        # pylint: disable=C0115,C0116
        return LlavaModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        from transformers import LlavaForConditionalGeneration

        source = LlavaForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        print(f"Converted Llava model to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted Llava model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target, image_newline=False):
        # pylint: disable=C0115,C0116
        mapping = {
            "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
            "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
            "language_model.lm_head.weight": "language_model.output_layer.weight",
        }
        if "vision_projection.encoder.linear_fc1.weight" in target.module.state_dict().keys():
            mapping.update(
                {
                    "multi_modal_projector.linear_1.weight": "vision_projection.encoder.linear_fc1.weight",
                    "multi_modal_projector.linear_1.bias": "vision_projection.encoder.linear_fc1.bias",
                    "multi_modal_projector.linear_2.weight": "vision_projection.encoder.linear_fc2.weight",
                    "multi_modal_projector.linear_2.bias": "vision_projection.encoder.linear_fc2.bias",
                }
            )
        elif "vision_projection.0.weight" in target.module.state_dict().keys():
            mapping.update(
                {
                    "multi_modal_projector.linear_1.weight": "vision_projection.0.weight",
                    "multi_modal_projector.linear_1.bias": "vision_projection.0.bias",
                    "multi_modal_projector.linear_2.weight": "vision_projection.2.weight",
                    "multi_modal_projector.linear_2.bias": "vision_projection.2.bias",
                }
            )
        else:
            raise KeyError("Unable to map vision projection keys.")

        if image_newline:
            mapping.update({"image_newline": "image_newline"})

        if "vision_model.vision_model.embeddings.class_embedding" in target.module.state_dict().keys():
            mapping.update(
                {
                    "vision_tower.vision_model.**": "vision_model.vision_model.**",
                }
            )
        elif "vision_model.class_token" in target.module.state_dict().keys():
            mapping.update(
                {
                    "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_model.conv1.weight",
                    "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_model.position_embeddings.weight",
                    "vision_tower.vision_model.encoder.layers.*.layer_norm1.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                    "vision_tower.vision_model.encoder.layers.*.layer_norm1.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
                    "vision_tower.vision_model.encoder.layers.*.layer_norm2.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                    "vision_tower.vision_model.encoder.layers.*.layer_norm2.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
                    "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
                    "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
                    "vision_tower.vision_model.encoder.layers.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
                    "vision_tower.vision_model.encoder.layers.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
                    "vision_tower.vision_model.encoder.layers.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
                    "vision_tower.vision_model.encoder.layers.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
                    "vision_tower.vision_model.pre_layrnorm.weight": "vision_model.ln_pre.weight",
                    "vision_tower.vision_model.pre_layrnorm.bias": "vision_model.ln_pre.bias",
                }
            )
        else:
            raise KeyError("Unable to map vision encoder keys.")
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _import_language_qkv,
                _import_vision_qkv,
                _import_vision_qkv_bias,
                _import_cls_token,
                _import_linear_fc1,
            ],
        )

    @property
    def tokenizer(self) -> "AutoTokenizer":
        # pylint: disable=C0115,C0116
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> LlavaConfig:
        # pylint: disable=C0115,C0116
        from transformers import LlavaConfig as HFLlavaConfig

        source = HFLlavaConfig.from_pretrained(str(self))
        text_conifg = source.text_config

        def make_vocab_size_divisible_by(vocab_size):
            # pylint: disable=C0115,C0116
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        language_transformer_config = LlamaConfig(
            num_layers=text_conifg.num_hidden_layers,
            hidden_size=text_conifg.hidden_size,
            ffn_hidden_size=text_conifg.intermediate_size,
            num_attention_heads=text_conifg.num_attention_heads,
            init_method_std=text_conifg.initializer_range,
            layernorm_epsilon=text_conifg.rms_norm_eps,
            num_query_groups=text_conifg.num_key_value_heads,
            rotary_base=text_conifg.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(text_conifg.vocab_size),
            share_embeddings_and_output_weights=False,
        )
        vision_transformer_config = HFCLIPVisionConfig(
            pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
        )
        vision_projection_config = MultimodalProjectorConfig(input_size=1024, hidden_size=4096, ffn_hidden_size=4096)

        output = LlavaConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            vision_projection_config=vision_projection_config,
            vision_feature_layer=source.vision_feature_layer,
        )

        return output


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


def export_qkv(linear_qkv, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
    # pylint: disable=C0115,C0116
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
    hidden_size = linear_qkv.size(-1)
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


def export_qkv_bias(qkv_bias: torch.Tensor, head_num, num_query_groups, heads_per_group, head_size):
    """
    Split interleave-concatenated qkv bias to separate q, k, v bias

    Example: export layer linear_qkv bias to HF {q|k|v}_proj bias
    """
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_bias = qkv_bias.reshape([qkv_total_dim, head_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_bias = qkv_bias[q_slice].reshape(-1).cpu()
    k_bias = qkv_bias[k_slice].reshape(-1).cpu()
    v_bias = qkv_bias[v_slice].reshape(-1).cpu()

    return q_bias, k_bias, v_bias


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.language_transformer_config
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


@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vision_qkv(ctx: io.TransformCTX, q, k, v):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config
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


@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_vision_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config
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
    source_key=("vision_tower.vision_model.embeddings.class_embedding",),
    target_key="vision_model.class_token",
)
def _import_cls_token(ctx: io.TransformCTX, cls_token):
    # pylint: disable=C0115,C0116
    return cls_token.reshape(1, 1, -1)


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.mlp.gate_proj.weight",
        "language_model.model.layers.*.mlp.up_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    # pylint: disable=C0115,C0116
    return torch.cat((down, gate), axis=0)
