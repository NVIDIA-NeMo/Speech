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
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
from torch import nn

from nemo.collections.llm.fn.activation import squared_relu
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.gpt.model.llama import _export_embedding, _export_head
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import NemotronConfig as HFNemotronConfig
    from transformers import NemotronForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class NemotronConfig(GPTConfig):
    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    activation_func: Callable = squared_relu
    position_embedding_type: str = "rope"
    share_embeddings_and_output_weights: bool = False
    add_bias_linear: bool = False

    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    rotary_percent: float = 0.5
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_add_fusion: bool = False
    layernorm_zero_centered_gamma: bool = True
    cross_entropy_loss_fusion: bool = True
    apply_rope_fusion: bool = True

    # Nemotron3Config4B as default configs
    num_layers: int = 32
    seq_length: int = 4096
    hidden_size: int = 3072
    ffn_hidden_size: int = 9216
    num_attention_heads: int = 24
    num_query_groups: Optional[int] = 8
    kv_channels: Optional[int] = 128
    init_method_std: float = 0.0134


@dataclass
class Nemotron3Config4B(NemotronConfig):
    num_layers: int = 32
    seq_length: int = 4096
    hidden_size: int = 3072
    ffn_hidden_size: int = 9216
    num_attention_heads: int = 24
    num_query_groups: int = 8
    kv_channels: Optional[int] = 128
    init_method_std: float = 0.0134


@dataclass
class Nemotron3Config8B(NemotronConfig):
    num_layers: int = 32
    seq_length: int = 4096
    hidden_size: int = 4096
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    num_query_groups: Optional[int] = None
    kv_channels: Optional[int] = None
    init_method_std: float = 0.010


@dataclass
class Nemotron3Config22B(NemotronConfig):
    num_layers: int = 40
    seq_length: int = 4096
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    num_query_groups: Optional[int] = None
    kv_channels: Optional[int] = None
    init_method_std: float = 0.008


@dataclass
class Nemotron4Config15B(NemotronConfig):
    num_layers: int = 32
    seq_length: int = 4096
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    num_query_groups: Optional[int] = 8
    kv_channels: Optional[int] = None
    init_method_std: float = 0.0134


@dataclass
class Nemotron4Config340B(NemotronConfig):
    num_layers: int = 96
    seq_length: int = 4096
    hidden_size: int = 18432
    ffn_hidden_size: int = 73728
    num_attention_heads: int = 96
    num_query_groups: Optional[int] = 8
    kv_channels: Optional[int] = None
    init_method_std: float = 0.0063


class NemotronModel(GPTModel):
    def __init__(
        self,
        config: Annotated[Optional[NemotronConfig], Config[NemotronConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or NemotronConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(NemotronModel, "hf")
class HFNemotronImporter(io.ModelConnector["NemotronForCausalLM", NemotronModel]):
    def init(self) -> NemotronModel:
        return NemotronModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import NemotronForCausalLM

        print('Start converting Nemotron model..')
        source = NemotronForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Nemotron model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.up_proj.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.input_layernorm.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "model.norm.bias": "decoder.final_layernorm.bias",
            "lm_head.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> NemotronConfig:
        from transformers import NemotronConfig as HFNemotronConfig

        source = HFNemotronConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = NemotronConfig(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            seq_length=source.max_position_embeddings,
            layernorm_epsilon=source.norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            rotary_percent=source.partial_rotary_factor,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(NemotronModel, "hf")
class HFNemotronExporter(io.ModelConnector[NemotronModel, "NemotronForCausalLM"]):
    def init(self, dtype=torch.bfloat16) -> "NemotronForCausalLM":
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return NemotronForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp.up_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.layers.*.input_layernorm.bias",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.layers.*.post_attention_layernorm.bias",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.final_layernorm.bias": "model.norm.bias",
        }

        return io.apply_transforms(
            source, target, mapping=mapping, transforms=[_export_qkv, _export_embedding, _export_head]
        )

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFNemotronConfig":
        from transformers import NemotronConfig as HFNemotronConfig

        source: NemotronConfig = io.load_context(str(self)).model.config

        return HFNemotronConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            partial_rotary_factor=source.rotary_percent,
            vocab_size=self.tokenizer.vocab_size,
        )


@io.state_transform(
    source_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

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
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_qkv(ctx: io.TransformCTX, linear_qkv):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, hidden_size])
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


__all__ = [
    "NemotronConfig",
    "Nemotron3Config4B",
    "Nemotron3Config8B",
    "Nemotron3Config22B",
    "Nemotron4Config15B",
    "Nemotron4Config340B",
    "NemotronModel",
]
