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

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer import (
    MegatronModule,
    ModuleSpec,
    TransformerConfig,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from torch import Tensor, nn

from nemo.collections.llm.fn.activation import openai_gelu
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils.import_utils import safe_import_from

if TYPE_CHECKING:
    from transformers import GemmaForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

TERowParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TERowParallelLinear")

TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")

TELayerNormColumnParallelLinear, _ = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TELayerNormColumnParallelLinear"
)


def gemma2_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """ """

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=Gemma2DotProductAttention,  # use unfused SDPA for attn logit softcapping
                    linear_proj=TERowParallelLinearLayerNorm,  # post attn RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinearLayerNorm,  # post mlp RMSNorm
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Note: Gemma requires huggingface transformers >= 4.38
# Note: these Gemma configs are copied from the corresponding HF model. You may need to modify the parameter for
# your own needs, in particular: seq_length and rotary_base.
@dataclass
class Gemma2Config(GPTConfig):
    """Gemma2 basic config"""

    # configs that are common across model sizes
    normalization: str = "RMSNorm"
    activation_func: Callable = openai_gelu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 8192
    kv_channels: int = 256
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = True
    layernorm_zero_centered_gamma: bool = True
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 10000
    window_size: tuple = (4096, 0)
    vocab_size: int = 256000
    gradient_accumulation_fusion: bool = False

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = gemma2_layer_spec
    # mcore customization
    query_pre_attn_scalar: int = 224
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0


@dataclass
class Gemma2Config2B(Gemma2Config):
    """Gemma2 2B config"""

    num_layers: int = 26
    hidden_size: int = 2304
    num_attention_heads: int = 8
    num_query_groups: int = 4
    ffn_hidden_size: int = 9216
    query_pre_attn_scalar: int = 256


@dataclass
class Gemma2Config9B(Gemma2Config):
    """Gemma2 9B config"""

    num_layers: int = 42
    hidden_size: int = 3584
    num_attention_heads: int = 16
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    query_pre_attn_scalar: int = 256


@dataclass
class Gemma2Config27B(Gemma2Config):
    """Gemma2 27B config"""

    num_layers: int = 46
    hidden_size: int = 4608
    num_attention_heads: int = 32
    num_query_groups: int = 16
    ffn_hidden_size: int = 36864
    query_pre_attn_scalar: int = 144


class Gemma2Model(GPTModel):
    """ """

    def __init__(
        self,
        config: Annotated[Optional[Gemma2Config], Config[Gemma2Config]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or Gemma2Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)

    def configure_model(self):
        from nemo.collections.common.parts.utils import extend_instance

        super().configure_model()
        if parallel_state.is_pipeline_first_stage(ignore_virtual=False):
            # Apply Embedding Scaling: sqrt(hidden_size)
            extend_instance(self.module.embedding, EmbeddingScalingMixin)
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False):
            # Prevents final logits from growing excessively by scaling them to a fixed range
            extend_instance(self.module.output_layer, Gemma2OutputLayer)


@io.model_importer(Gemma2Model, "hf")
class HFGemmaImporter(io.ModelConnector["GemmaForCausalLM", Gemma2Model]):
    """ """

    def init(self) -> Gemma2Model:
        return Gemma2Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import Gemma2ForCausalLM

        source = Gemma2ForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()

        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Gemma2 model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """ """
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.pre_feedforward_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.post_feedforward_layernorm.weight": (
                "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight"
            ),
            "model.layers.*.post_attention_layernorm.weight": (
                "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight"
            ),
            "model.norm.weight": "decoder.final_layernorm.weight",
        }

        transforms = [
            io.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            ),
            io.state_transform(
                source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """ """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> Gemma2Config:
        """ """
        from transformers import GemmaConfig as HFGemmaConfig
        from transformers import GenerationConfig

        source = HFGemmaConfig.from_pretrained(str(self))
        generation_config = GenerationConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = Gemma2Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            query_pre_attn_scalar=source.query_pre_attn_scalar,
            attn_logit_softcapping=source.attn_logit_softcapping,
            final_logit_softcapping=source.final_logit_softcapping,
            window_size=(source.sliding_window, 0),
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            vocab_size=source.vocab_size,
            share_embeddings_and_output_weights=True,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
        )

        return output


@io.model_exporter(Gemma2Model, "hf")
class HFGemmaExporter(io.ModelConnector[Gemma2Model, "GemmaForCausalLM"]):
    """ """

    def init(self) -> "GemmaForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        """ """

        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        """ """

        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm.weight"
            ),
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self):
        """ """

        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "Gemma2Config":
        """ """

        source: Gemma2Config = io.load_context(str(self), subpath="model.config")

        from transformers import Gemma2Config as HFGemmaConfig

        return HFGemmaConfig(
            architectures=["Gemma2ForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            vocab_size=self.tokenizer.vocab_size,
            rope_theta=source.rotary_base,
            query_pre_attn_scalar=source.query_pre_attn_scalar,
            attn_logit_softcapping=source.attn_logit_softcapping,
            final_logit_softcapping=source.final_logit_softcapping,
        )


class Gemma2DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        **kwargs,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)

        self.window_size = None
        if self.layer_number % 2 == 0:
            self.window_size = config.window_size

        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        self.norm_factor = math.sqrt(config.query_pre_attn_scalar)

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ):
        """Forward.
        Modified from mcore.transformer.dot_product_attention to support Gemma2-specific
        final_logit_softcapping.
        """
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention." "Please use TEDotProductAttention instead."
        )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query.dtype,
            "mpu",
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )
        # Gemma 2 specific:
        matmul_result = logit_softcapping(matmul_result, self.config.attn_logit_softcapping)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # sliding window attention
        if attention_mask is not None and self.window_size is not None:
            attention_mask = get_swa(query.size(0), key.size(0), self.window_size)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)
        return context


class TERowParallelLinearLayerNorm(TERowParallelLinear):
    """Modified From TERowParallelLinear with an additional Post-LN."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            config=config,
            **kwargs,
        )
        self.post_layernorm = TENorm(config, output_size)

    def forward(self, x):
        """Forward with additional Post LN on output"""
        output, bias = super().forward(x)
        return self.post_layernorm(output), bias


class Gemma2OutputLayer(ColumnParallelLinear):
    """Extends from ColumnParallelLinear with logit soft capping."""

    def forward(self, *args, **kwargs):
        """Forward with logit soft capping."""
        output, bias = super().forward(*args, **kwargs)
        output = logit_softcapping(output, self.config.final_logit_softcapping)
        return output, bias


class EmbeddingScalingMixin(torch.nn.Module):
    """
    A mixin class for scaling embeddings in Megatron GPT.
    The scaling is applied only if the configuration (accessible via `self.config`)
    includes `apply_embedding_scaling` set to True.
    """

    def forward(self, **kwargs):
        """
        Forward pass that scales the output embeddings from the `forward` method of
        the superclass by the square root of the hidden size specified in the configuration.
        """
        embeddings = super().forward(**kwargs)
        return embeddings * torch.tensor(self.config.hidden_size**0.5, dtype=embeddings.dtype)


def logit_softcapping(logits: torch.Tensor, scale: Optional[float]):
    """Prevents logits from growing excessively by scaling them to a fixed range"""
    if not scale:
        return logits

    return scale * torch.tanh(logits / scale)


def get_swa(seq_q, seq_kv, w):
    """Create the equivalent attention mask fro SWA in [seq_q, seq_kv] shape"""
    m = torch.ones(seq_q, seq_kv, dtype=torch.bool, device="cuda")
    mu = torch.triu(m, diagonal=seq_kv - seq_q - w[0])
    ml = torch.tril(mu, diagonal=seq_kv - seq_q + w[1])
    ml = ~ml

    return ml


__all__ = [
    "Gemma2Config",
    "Gemma2Config2B",
    "Gemma2Config9B",
    "Gemma2Config27B",
    "Gemma2Model",
]
