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
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnBackend
from torch import nn

from nemo.collections.llm.fn.activation import openai_gelu
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import GemmaForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


# Note: Gemma requires huggingface transformers >= 4.38
# Note: these Gemma configs are copied from the corresponding HF model. You may need to modify the parameter for
# your own needs, in particular: seq_length and rotary_base.
@dataclass
class GemmaConfig(GPTConfig):
    """Gemma basic config"""

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
    # Note: different behavior compared to Legacy NeMo
    # Legacy NeMo does not set layernorm_zero_centered_gamma and instead adds 1 in the HF -> NeMo conversion script
    # The present implementation is more in line with the official implementation
    layernorm_zero_centered_gamma: bool = True
    # Disable cuDNN attention since TE 1.8 does not support head dim > 128
    attention_backend: AttnBackend = AttnBackend.flash


@dataclass
class GemmaConfig2B(GemmaConfig):
    """Gemma 2B config"""

    num_layers: int = 18
    hidden_size: int = 2048
    num_attention_heads: int = 8
    num_query_groups: int = 1
    ffn_hidden_size: int = 16384


@dataclass
class GemmaConfig7B(GemmaConfig):
    """Gemma 7B config"""

    num_layers: int = 28
    hidden_size: int = 3072
    num_attention_heads: int = 16
    num_query_groups: int = 16
    ffn_hidden_size: int = 24576


class CodeGemmaConfig2B(GemmaConfig2B):
    """Code Gemma 2B config"""

    pass


class CodeGemmaConfig7B(GemmaConfig7B):
    """Code Gemma 7B config"""

    pass


class GemmaModel(GPTModel):
    """ """

    def __init__(
        self,
        config: Annotated[Optional[GemmaConfig], Config[GemmaConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        """ """
        super().__init__(config or GemmaConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)

    def configure_model(self):
        """ """
        from nemo.collections.common.parts.utils import extend_instance
        from nemo.collections.llm.gpt.model.gemma2 import EmbeddingScalingMixin

        super().configure_model()
        if parallel_state.is_pipeline_first_stage(ignore_virtual=False):
            extend_instance(self.module.embedding, EmbeddingScalingMixin)


@io.model_importer(GemmaModel, "hf")
class HFGemmaImporter(io.ModelConnector["GemmaForCausalLM", GemmaModel]):
    """ """

    def init(self) -> GemmaModel:
        """ """
        return GemmaModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """ """
        from transformers import GemmaForCausalLM

        source = GemmaForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Gemma model to Nemo, model saved to {output_path}")

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
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
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
    def config(self) -> GemmaConfig:
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

        output = GemmaConfig(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=True,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
        )

        return output


@io.model_exporter(GemmaModel, "hf")
class HFGemmaExporter(io.ModelConnector[GemmaModel, "GemmaForCausalLM"]):
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
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
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
    def config(self) -> "GemmaConfig":
        """ """
        source: GemmaConfig = io.load_context(str(self), subpath="model.config")

        from transformers import GemmaConfig as HFGemmaConfig

        return HFGemmaConfig(
            architectures=["GemmaForCausalLM"],
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
        )


__all__ = [
    "GemmaConfig",
    "GemmaConfig2B",
    "GemmaConfig7B",
    "CodeGemmaConfig2B",
    "CodeGemmaConfig7B",
    "GemmaModel",
]
