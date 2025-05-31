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

"""Gemma3 language model"""

import copy
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional, Tuple, Union

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import ModuleSpec, TransformerConfig, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from torch import Tensor, nn

from nemo.collections.llm.fn.activation import openai_gelu
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.gpt.model.gemma2 import TERowParallelLinearLayerNorm
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils.import_utils import safe_import_from

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from transformers import Gemma3ForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

TERowParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TERowParallelLinear")

TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")

TELayerNormColumnParallelLinear, _ = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TELayerNormColumnParallelLinear"
)

TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")


"""
The special design of gemma3:

- Special RMSNorm
  x * (1 + w) instead of x * w
  (x * w).to(dtype) instead of x.to(dtype) * w

- Post attention norm

- Post MLP norm

- Post word embedding scaling

- The 27B model sets custom q_scaling as 168^(-0.5), others use default head_dim^(-0.5)

- Interleaved attention layers
  Pattern: 5 local layers + 1 global layers

- Global layer and local layer calculate rope embedding differently
  rope_base:
    local: 10_000
    global: 1_000_000
  rope_scaling (linear):
    local: 1.0
    global: 8.0

vision:
- Post vision encoding norm
- Single layer linear vision projection
"""


def gemma3_layer_spec(config: GPTConfig) -> ModuleSpec:
    """Gemma3 custom layer spec."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Gemma3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=Gemma3TEDotProductAttention,  # mixed gloabl/local attn
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    linear_proj=TERowParallelLinearLayerNorm,  # post attn RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,  # residual link
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinearLayerNorm,  # post mlp RMSNorm
                ),
            ),
            mlp_bda=get_bias_dropout_add,  # residual link
        ),
    )


@dataclass
class Gemma3Config(GPTConfig):
    """Gemma3 basic config"""

    seq_length: int = 131_072

    # embedding
    vocab_size: int = 262_208  # Gemma3 1B model has smaller embedding table
    position_embedding_type: str = "rope"
    rotary_base: tuple = (10_000, 1_000_000)  # (local, global)
    share_embeddings_and_output_weights: bool = True

    # norm
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = True  # x * (1 + w)
    layernorm_epsilon: float = 1e-6

    # attention
    window_size: tuple = 512  # local
    interleaved_attn_pattern: tuple = (5, 1)  # (local, global)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # mlp
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = openai_gelu

    # Do not change
    is_vision_language: bool = False
    flash_decode: bool = False
    gradient_accumulation_fusion: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = gemma3_layer_spec
    scatter_embedding_sequence_parallel: bool = True

    def configure_model(
        self,
        tokenizer,
        pre_process=None,
        post_process=None,
        vp_stage: Optional[int] = None,
    ) -> "MCoreGPTModel":
        """Configure and instantiate a megatron-core Gemma3 model."""
        if self.context_parallel_size > 1:
            raise ValueError("Context Parallel is not supported for Gemma3 model.")

        assert (
            getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None
        ), "Virtual pipeline model parallel size is not yet supported for Gemma3 model."

        rotary_base_local, rotary_base_global = self.rotary_base
        # Trick megatron's RotaryEmbedding to initialize the model successfully
        self.rotary_base = rotary_base_global
        model = super().configure_model(tokenizer, pre_process, post_process, vp_stage=vp_stage)
        self.rotary_base = (rotary_base_local, rotary_base_global)

        # Replace model's embedding and rope with customized ones
        if hasattr(model, 'embedding'):
            model.embedding = Gemma3LanguageModelEmbedding(
                config=self,
                vocab_size=self.vocab_size,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )
        model.rotary_pos_emb = Gemma3RotaryEmbedding(
            kv_channels=self.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=rotary_base_global,
            rope_scaling=False,
            rope_scaling_factor=self.rope_scaling_factor,
            use_cpu_initialization=self.use_cpu_initialization,
            rotary_base_local=rotary_base_local,
        )
        if hasattr(model, 'embedding') or hasattr(model, 'output_layer'):
            model.setup_embeddings_and_output_layer()
        return model


@dataclass
class Gemma3Config1B(Gemma3Config):
    """Gemma3 1B config"""

    is_vision_language: bool = False
    num_layers: int = 26
    hidden_size: int = 1152
    num_attention_heads: int = 4
    num_query_groups: int = 1
    kv_channels: int = 256
    ffn_hidden_size: int = 6912
    window_size: int = 512
    rope_scaling_factor: float = 1.0  # no rope scaling
    seq_length: int = 32768
    bf16: bool = True
    vocab_size: int = 262_144


@dataclass
class Gemma3Config4B(Gemma3Config):
    """Gemma3 4B config"""

    is_vision_language: bool = True
    num_layers: int = 34
    hidden_size: int = 2560
    num_attention_heads: int = 8
    num_query_groups: int = 4
    kv_channels: int = 256
    ffn_hidden_size: int = 10240
    window_size: int = 1024
    rope_scaling_factor: float = 8.0


@dataclass
class Gemma3Config12B(Gemma3Config):
    """Gemma3 12B config"""

    is_vision_language: bool = True
    num_layers: int = 48
    hidden_size: int = 3840
    num_attention_heads: int = 16
    num_query_groups: int = 8
    kv_channels: int = 256
    ffn_hidden_size: int = 15360
    window_size: int = 1024
    rope_scaling_factor: float = 8.0


@dataclass
class Gemma3Config27B(Gemma3Config):
    """Gemma3 27B config"""

    is_vision_language: bool = True
    num_layers: int = 62
    hidden_size: int = 5376
    num_attention_heads: int = 32
    num_query_groups: int = 16
    kv_channels: int = 128
    softmax_scale: int = 1.0 / math.sqrt(168)  # only for 27B, (5376 // 32)^(-0.5)
    ffn_hidden_size: int = 21504
    window_size: int = 1024
    rope_scaling_factor: float = 8.0


class Gemma3Model(GPTModel):
    """Gemma3 base model"""

    def __init__(
        self,
        config: Annotated[Optional[Gemma3Config], Config[Gemma3Config]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
        model_context_managers: Optional[list] = [],
    ):
        super().__init__(
            config or Gemma3Config(),
            optim=optim,
            tokenizer=tokenizer,
            model_transform=model_transform,
            model_context_managers=model_context_managers,
        )


class Gemma3LanguageModelEmbedding(LanguageModelEmbedding):
    """Gemma3 language token embedding.

    Adds a normalization to the embedding.
    """

    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None) -> Tensor:
        """Calculate embedding and normalize"""
        embeddings = super().forward(input_ids, position_ids, tokentype_ids)
        embeddings = embeddings * (self.config.hidden_size**0.5)
        return embeddings


class Gemma3RotaryEmbedding(RotaryEmbedding):
    """Gemma3 position rope embedding.

    Calculates rope embeddings for both local and global attention layers.
    """

    def __init__(
        self,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        rotary_base: int = 1_000_000,
        rotary_base_local: int = 10_000,
        **kwargs,
    ):
        # The rope scaling in RotaryEmbedding is not linear scaling,
        # so this flag must be off. Will calculate linear scaling below.
        assert rope_scaling is False

        # Get inv_freq for global attention layers
        super().__init__(
            rope_scaling=rope_scaling,
            rotary_base=rotary_base,
            **kwargs,
        )
        self.inv_freq /= rope_scaling_factor

        # Setup Rotary Embedding for local attentions
        self.rope_local = RotaryEmbedding(
            rope_scaling=rope_scaling,
            rotary_base=rotary_base_local,
            **kwargs,
        )

    @lru_cache(maxsize=32)
    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        """Get global and local rope embedding"""
        rope_global = super().forward(max_seq_len, offset, packed_seq)
        rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq)
        return rope_local, rope_global


def _is_local_attn_layer(
    layer_number: int,
    layer_pattern: Tuple[int, int],
) -> bool:
    pattern_size = sum(layer_pattern)
    return layer_number % pattern_size != 0


class Gemma3SelfAttention(SelfAttention):
    """Gemma3 self attention.

    Uses local rope embedding for local layers,
    global rope embedding for global layers.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Switch to either local or global rope embedding before forward"""
        assert isinstance(rotary_pos_emb, tuple)
        assert rotary_pos_cos is None and rotary_pos_sin is None

        if _is_local_attn_layer(self.layer_number, self.config.interleaved_attn_pattern):
            final_rotary_pos_emb = rotary_pos_emb[0]
        else:
            final_rotary_pos_emb = rotary_pos_emb[1]
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=final_rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )


class Gemma3TEDotProductAttention(TEDotProductAttention):
    """Gemma3 core attention.

    Switches between global and local sliding window attention
    based on the layer_number and pre-defined layer pattern.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ):
        # Overwrite config.window_size based on layer_number
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.interleaved_attn_pattern):
            # local attention, (q, k)
            config.window_size = (config.window_size, 0)
        else:
            # global attention
            config.window_size = None

        # The VL model calculates mask manually
        if config.is_vision_language:
            attn_mask_type = AttnMaskType.arbitrary

        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )


@io.model_importer(Gemma3Model, "hf")
class HFGemma3Importer(io.ModelConnector["Gemma3ForCausalLM", Gemma3Model]):
    """Gemma3 Huggingface importer"""

    def init(self) -> Gemma3Model:
        return Gemma3Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import Gemma3ForCausalLM

        source = Gemma3ForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()

        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted HF Gemma3 model to Nemo, saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116
        mapping = {
            # word emebdding
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            # attention
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.self_attn.q_norm.weight": "decoder.layers.*.self_attention.q_layernorm.weight",
            "model.layers.*.self_attn.k_norm.weight": "decoder.layers.*.self_attention.k_layernorm.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.post_attention_layernorm.weight": (
                "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight"
            ),
            # mlp
            "model.layers.*.pre_feedforward_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.post_feedforward_layernorm.weight": (
                "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight"
            ),
            # final norm
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
        # pylint: disable=C0115,C0116
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> Gemma3Config:
        # pylint: disable=C0115,C0116
        from transformers import Gemma3Config as HFGemma3Config
        from transformers import GenerationConfig

        name = str(self)
        source = HFGemma3Config.from_pretrained(name)
        source_text = source.text_config
        generation_config = GenerationConfig.from_pretrained(name)

        if source_text.num_hidden_layers == 26:
            output = Gemma3Config1B()
        elif source_text.num_hidden_layers == 34:
            output = Gemma3Config4B()
        elif source_text.num_hidden_layers == 48:
            output = Gemma3Config12B()
        elif source_text.num_hidden_layers == 62:
            output = Gemma3Config27B()
        else:
            raise ValueError(f"Unrecognized import model: {name}")

        output.params_dtype = dtype_from_hf(source)
        output.init_method_std = source.initializer_range
        output.generation_config = generation_config

        return output


@io.model_exporter(Gemma3Model, "hf")
class HFGemma3Exporter(io.ModelConnector[Gemma3Model, "Gemma3ForCausalLM"]):
    """Export Gemma3 to HF format"""

    def init(self) -> "Gemma3ForCausalLM":
        from transformers import Gemma3ForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return Gemma3ForCausalLM._from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116
        mapping = {
            # word emebdding
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # mlp
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm.weight"
            ),
            # final norm
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
        # pylint: disable=C0115,C0116
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self):
        # pylint: disable=C0115,C0116
        source: Gemma3Config = io.load_context(str(self), subpath="model.config")

        from transformers import Gemma3TextConfig as HFGemma3TextConfig

        output = HFGemma3TextConfig(
            architectures=["Gemma3ForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=source.kv_channels,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            vocab_size=source.vocab_size,
            rope_theta=source.rotary_base[1],
            rope_local_base_freq=source.rotary_base[0],
        )
        if source.num_layers == 62:  # 27B
            output.query_pre_attn_scalar = 168
        else:
            output.query_pre_attn_scalar = output.head_dim
        return output


__all__ = [
    "Gemma3Config",
    "Gemma3Config1B",
    "Gemma3Config4B",
    "Gemma3Config12B",
    "Gemma3Config27B",
    "Gemma3Model",
]
