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
from typing import TYPE_CHECKING, Annotated, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import Starcoder2Config as HFStarcoder2Config
    from transformers import Starcoder2ForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class Starcoder2Config(GPTConfig):
    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    activation_func: Callable = F.gelu
    add_bias_linear: bool = True
    seq_length: int = 16384
    position_embedding_type: str = "rope"
    rotary_percent: float = 1.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    init_method_std: float = 0.01
    share_embeddings_and_output_weights: bool = False
    kv_channels: int = None
    num_query_groups: int = None
    window_size: Optional[List[int]] = None
    attention_softmax_in_fp32: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    layernorm_epsilon: float = 1e-5


@dataclass
class Starcoder2Config3B(Starcoder2Config):
    num_layers: int = 30
    hidden_size: int = 3072
    ffn_hidden_size: int = 12288
    num_query_groups: int = 2
    num_attention_heads: int = 24
    init_method_std: float = 0.018042
    rotary_base: float = 999999.4420358813


@dataclass
class Starcoder2Config7B(Starcoder2Config):
    num_layers: int = 32
    hidden_size: int = 4608
    ffn_hidden_size: int = 18432
    num_query_groups: int = 4
    num_attention_heads: int = 36
    init_method_std: float = 0.018042
    rotary_base: float = 1_000_000


@dataclass
class Starcoder2Config15B(Starcoder2Config):
    num_layers: int = 40
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_query_groups: int = 4
    num_attention_heads: int = 48
    init_method_std: float = 0.01275
    rotary_base: float = 100_000


class Starcoder2Model(GPTModel):
    def __init__(
        self,
        config: Annotated[Optional[Starcoder2Config], Config[Starcoder2Config]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or Starcoder2Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(Starcoder2Model, "hf")
class HFStarcoder2Importer(io.ModelConnector["Starcoder2ForCausalLM", Starcoder2Model]):
    def init(self) -> Starcoder2Model:
        return Starcoder2Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import Starcoder2ForCausalLM

        source = Starcoder2ForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Starcoder2 model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.o_proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            "model.layers.*.mlp.c_fc.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "model.layers.*.mlp.c_fc.bias": "decoder.layers.*.mlp.linear_fc1.bias",
            "model.layers.*.mlp.c_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.mlp.c_proj.bias": "decoder.layers.*.mlp.linear_fc2.bias",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.input_layernorm.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "model.norm.bias": "decoder.final_layernorm.bias",
            "lm_head.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv_bias, _import_qkv_weight])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> Starcoder2Config:
        from transformers import Starcoder2Config as HFStarcoder2Config

        source = HFStarcoder2Config.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = Starcoder2Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            seq_length=source.max_position_embeddings,
            layernorm_epsilon=source.norm_epsilon,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(Starcoder2Model, "hf")
class HFStarcoder2Exporter(io.ModelConnector[Starcoder2Model, "Starcoder2ForCausalLM"]):
    def init(self) -> "Starcoder2ForCausalLM":
        from transformers import Starcoder2ForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return Starcoder2ForCausalLM._from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.bias": "model.layers.*.self_attn.o_proj.bias",
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp.c_fc.weight",
            "decoder.layers.*.mlp.linear_fc1.bias": "model.layers.*.mlp.c_fc.bias",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.c_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.bias": "model.layers.*.mlp.c_proj.bias",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.layers.*.input_layernorm.bias",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.layers.*.post_attention_layernorm.bias",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.final_layernorm.bias": "model.norm.bias",
            "output_layer.weight": "lm_head.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_export_qkv_weight, _export_qkv_bias])

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFStarcoder2Config":
        from transformers import Starcoder2Config as HFStarcoder2Config

        source: Starcoder2Config = io.load_context(str(self)).model.config

        return HFStarcoder2Config(
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
def _import_qkv_weight(ctx: io.TransformCTX, q, k, v):
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
    source_key=(
        "model.layers.*.self_attn.q_proj.bias",
        "model.layers.*.self_attn.k_proj.bias",
        "model.layers.*.self_attn.v_proj.bias",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias(ctx: io.TransformCTX, qb, kb, vb):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = megatron_config.kv_channels

    new_q_bias_tensor_shape = (head_num, head_size)
    new_kv_bias_tensor_shape = (num_query_groups, head_size)

    qb = qb.view(*new_q_bias_tensor_shape)
    kb = kb.view(*new_kv_bias_tensor_shape)
    vb = vb.view(*new_kv_bias_tensor_shape)

    qkv_bias_l = []
    for i in range(num_query_groups):
        qkv_bias_l.append(qb[i * heads_per_group : (i + 1) * heads_per_group, :])
        qkv_bias_l.append(kb[i : i + 1, :])
        qkv_bias_l.append(vb[i : i + 1, :])

    qkv_bias = torch.cat(qkv_bias_l)
    qkv_bias = qkv_bias.reshape([head_size * (head_num + 2 * num_query_groups)])

    return qkv_bias


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_qkv_weight(ctx: io.TransformCTX, linear_qkv):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_num = megatron_config.num_attention_heads
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


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.bias",
    target_key=(
        "model.layers.*.self_attn.q_proj.bias",
        "model.layers.*.self_attn.k_proj.bias",
        "model.layers.*.self_attn.v_proj.bias",
    ),
)
def _export_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = megatron_config.kv_channels
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
