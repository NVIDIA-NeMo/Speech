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
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import AutoConfig, AutoModelForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class ChatGLMConfig(GPTConfig):
    num_layers: int = 28
    hidden_size: int = 4096
    ffn_hidden_size: int = 13696
    num_attention_heads: int = 32
    num_query_groups: int = 2
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    rotary_percent: float = 0.5
    rotary_interleaved: bool = True
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    share_embeddings_and_output_weights: bool = False
    make_vocab_size_divisible_by: int = 65024  # override vocab size


@dataclass
class ChatGLM2Config6B(ChatGLMConfig):
    seq_length: int = 32768


@dataclass
class ChatGLM3Config6B(ChatGLMConfig):
    seq_length: int = 8192


class ChatGLMModel(GPTModel):
    def __init__(
        self,
        config: Annotated[Optional[ChatGLMConfig], Config[ChatGLMConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or ChatGLMConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(ChatGLMModel, "hf")
class HFChatGLMImporter(io.ModelConnector["AutoModelForCausalLM", ChatGLMModel]):
    def init(self) -> ChatGLMModel:
        return ChatGLMModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import AutoModelForCausalLM

        source = AutoModelForCausalLM.from_pretrained(str(self), trust_remote_code=True, torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted ChatGLM model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "transformer.embedding.word_embeddings.weight": "embedding.word_embeddings.weight",
            "transformer.encoder.layers.*.self_attention.dense.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "transformer.encoder.layers.*.mlp.dense_h_to_4h.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "transformer.encoder.layers.*.mlp.dense_4h_to_h.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "transformer.encoder.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "transformer.encoder.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "transformer.encoder.final_layernorm.weight": "decoder.final_layernorm.weight",
            "transformer.output_layer.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv_weight, _import_qkv_bias])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)), trust_remote_code=True)

    @property
    def config(self) -> ChatGLMConfig:
        from transformers import AutoConfig as HFAutoConfig

        source = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        output = ChatGLMConfig(
            num_layers=source.num_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            seq_length=source.seq_length,
            num_query_groups=source.multi_query_group_num,
            make_vocab_size_divisible_by=source.padded_vocab_size,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(ChatGLMModel, "hf")
class HFChatGLMExporter(io.ModelConnector[ChatGLMModel, "AutoModelForCausalLM"]):
    def init(self) -> "AutoModelForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)

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
            "embedding.word_embeddings.weight": "transformer.embedding.word_embeddings.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "transformer.encoder.layers.*.self_attention.dense.weight",
            "decoder.layers.*.mlp.linear_fc1.weight": "transformer.encoder.layers.*.mlp.dense_h_to_4h.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "transformer.encoder.layers.*.mlp.dense_4h_to_h.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "transformer.encoder.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "transformer.encoder.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "transformer.encoder.final_layernorm.weight",
            "output_layer.weight": "transformer.output_layer.weight",
        }

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _export_qkv_weight,
                _export_qkv_bias,
            ],
        )

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "AutoConfig":
        source: ChatGLMConfig = io.load_context(str(self)).model.config

        return AutoConfig(
            num_layers=source.num_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            seq_length=source.seq_length,
            multi_query_group_num=source.num_query_groups,
            padded_vocab_size=self.tokenizer.vocab_size,
        )


@io.state_transform(
    source_key="transformer.encoder.layers.*.self_attention.query_key_value.weight",
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv_weight(ctx: io.TransformCTX, hf_qkv_weights):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = hf_qkv_weights.size()
    new_q_tensor_shape = (head_num, head_size, old_tensor_shape[1])
    new_kv_tensor_shape = (num_query_groups, head_size, old_tensor_shape[1])
    q, k, v = hf_qkv_weights.split(
        [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size], dim=0
    )
    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights = torch.empty((0, head_size, old_tensor_shape[1]))
    for i in range(num_query_groups):
        qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key="transformer.encoder.layers.*.self_attention.query_key_value.bias",
    target_key="decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias(ctx: io.TransformCTX, hf_qkv_bias):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = megatron_config.kv_channels

    new_q_tensor_shape = (head_num, head_size)
    new_kv_tensor_shape = (num_query_groups, head_size)
    q, k, v = hf_qkv_bias.split(
        [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size], dim=0
    )
    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)
    qkv_bias = torch.empty((0, head_size))
    for i in range(num_query_groups):
        qkv_bias = torch.cat((qkv_bias, q[i * heads_per_group : (i + 1) * heads_per_group, :]))
        qkv_bias = torch.cat((qkv_bias, k[i : i + 1, :]))
        qkv_bias = torch.cat((qkv_bias, v[i : i + 1, :]))
    qkv_bias = qkv_bias.reshape(
        [
            head_size * (head_num + 2 * num_query_groups),
        ]
    )
    return qkv_bias


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key="transformer.encoder.layers.*.self_attention.query_key_value.weight",
)
def _export_qkv_weight(ctx: io.TransformCTX, qkv_weights):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_weight = qkv_weights[q_slice].reshape(-1, hidden_size)
    k_weight = qkv_weights[k_slice].reshape(-1, hidden_size)
    v_weight = qkv_weights[v_slice].reshape(-1, hidden_size)
    return torch.cat((q_weight, k_weight, v_weight), dim=0)


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.bias",
    target_key="transformer.encoder.layers.*.self_attention.query_key_value.bias",
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

    q_bias = qkv_bias[q_slice].reshape(
        -1,
    )
    k_bias = qkv_bias[k_slice].reshape(
        -1,
    )
    v_bias = qkv_bias[v_slice].reshape(
        -1,
    )
    return torch.cat((q_bias, k_bias, v_bias))


__all__ = [
    "ChatGLMConfig",
    "ChatGLM2Config6B",
    "ChatGLM3Config6B",
    "ChatGLMModel",
]
