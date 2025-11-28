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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import Phi3Config as HFPhi3Config
    from transformers import Phi3ForCausalLM

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class Phi3Config(GPTConfig):
    # pylint: disable=C0115,C0116
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False


@dataclass
class Phi3ConfigMini(Phi3Config):
    # pylint: disable=C0115,C0116
    num_layers: int = 32
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32
    num_query_groups: int = 32
    rotary_base: float = 10000.0
    vocab_size: int = 32064


class Phi3Model(GPTModel):
    # pylint: disable=C0115,C0116
    def __init__(
        self,
        config: Optional[Phi3Config] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or Phi3Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(Phi3Model, "hf")
class HFPhi3Importer(io.ModelConnector["Phi3ForCausalLM", Phi3Model]):
    # pylint: disable=C0115,C0116
    def init(self) -> Phi3Model:
        return Phi3Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import Phi3ForCausalLM

        # Check if the source is valid model identifier or path
        try:
            source = Phi3ForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        except Exception as e:
            raise ValueError(f"Failed to load the model from source '{self}': {e}")

        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Phi3 model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116
        # Define mapping for mini-4k-instruct
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.qkv_proj.weight": "decoder.layers.*.self_attention.linear_qkv.weight",
            "model.layers.*.mlp.gate_up_proj.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv])

    @property
    def tokenizer(self):
        # pylint: disable=C0115,C0116
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> Phi3Config:
        # pylint: disable=C0115,C0116
        from transformers import Phi3Config as HFPhi3Config

        source = HFPhi3Config.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = Phi3Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )
        print("output:", output)
        return output


@io.model_exporter(Phi3Model, "hf")
class HFPhi3Exporter(io.ModelConnector[Phi3Model, "Phi3ForCausalLM"]):
    # pylint: disable=C0115,C0116
    def init(self) -> "Phi3ForCausalLM":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        target.cpu().save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp.gate_up_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_export_qkv])

    @property
    def tokenizer(self):
        # pylint: disable=C0115,C0116
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFPhi3Config":
        # pylint: disable=C0115,C0116
        source: Phi3Config = io.load_context(str(self), subpath="model.config")

        from transformers import Phi3Config as HFPhi3Config

        return HFPhi3Config(
            architectures=["Phi3ForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=0.02,
            rms_norm_eps=1e-05,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )


@io.state_transform(
    source_key="model.layers.*.self_attn.qkv_proj.weight",
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, qkv_weight):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = qkv_weight.size()
    new_q_tensor_shape = (head_num, head_size, old_tensor_shape[1])
    new_kv_tensor_shape = (num_query_groups, head_size, old_tensor_shape[1])
    q, k, v = qkv_weight.split(
        [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size], dim=0
    )
    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights = torch.empty((0, head_size, old_tensor_shape[1])).type_as(qkv_weight)
    for i in range(num_query_groups):
        qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key="model.layers.*.self_attn.qkv_proj.weight",
)
def _export_qkv(ctx: io.TransformCTX, linear_qkv):
    """Transform function to convert fused QKV weights to separate Q,K,V format.

    Converts NeMo's fused QKV projection weights to HF's separate Q, K, V format,
    handling grouped query attention (GQA) appropriately.

    Args:
        ctx: Transform context
        linear_qkv: Fused QKV projection weights

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Separate Q, K, V projection weights
    """
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

    return torch.cat([q_proj, k_proj, v_proj], dim=0)


__all__ = ["Phi3Config", "Phi3ConfigMini", "Phi3Model"]
