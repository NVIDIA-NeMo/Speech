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
from typing import TYPE_CHECKING, Annotated, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import Starcoder2Config as HFStarcoder2Config
    from transformers import Starcoder2ForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class Starcoder2Config(GPTConfig):
    """
    Configuration class for the Starcoder2 Config, inheriting from GPTConfig.
    """

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
    """
    Configuration class for the Starcoder2 3B Config, inheriting from Starcoder2Config.
    """

    num_layers: int = 30
    hidden_size: int = 3072
    ffn_hidden_size: int = 12288
    num_query_groups: int = 2
    num_attention_heads: int = 24
    init_method_std: float = 0.018042
    rotary_base: float = 999999.4420358813


@dataclass
class Starcoder2Config7B(Starcoder2Config):
    """
    Configuration class for the Starcoder2 7B Config, inheriting from Starcoder2Config.
    """

    num_layers: int = 32
    hidden_size: int = 4608
    ffn_hidden_size: int = 18432
    num_query_groups: int = 4
    num_attention_heads: int = 36
    init_method_std: float = 0.018042
    rotary_base: float = 1_000_000


@dataclass
class Starcoder2Config15B(Starcoder2Config):
    """
    Configuration class for the Starcoder2 15B Config, inheriting from Starcoder2Config.
    """

    num_layers: int = 40
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_query_groups: int = 4
    num_attention_heads: int = 48
    init_method_std: float = 0.01275
    rotary_base: float = 100_000


class Starcoder2Model(GPTModel):
    """
    Starcoder2 model implementation based on the GPT model architecture.

    This class provides a high-level interface for Starcoder2 models,
    implementing the specific architecture and settings needed for Starcoder2 models.
    """

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
    """
    Importer for converting Hugging Face Starcoder2 models to NeMo format.

    This class handles the conversion of Hugging Face's Starcoder2ForCausalLM models
    to NeMo's Starcoder2 format, including weight mapping and configuration translation.
    """

    def init(self) -> Starcoder2Model:
        """
        Initialize a NeMo Starcoder2Model instance.

        Returns:
            Starcoder2Model: Initialized NeMo Starcoder2 model with the appropriate configuration
                        and tokenizer.
        """
        return Starcoder2Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """
        Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
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
        """
        Convert state dict from HF format to NeMo format.

        Maps the weights from the HF model to the NeMo model according to
        the appropriate mapping scheme.

        Args:
            source: Source HF model
            target: Target NeMo model

        Returns:
            The result of applying the transforms
        """
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
                source_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.bias",
                fn=TransformFns.merge_qkv_bias,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """
        Get the tokenizer for the HF model.

        Returns:
            AutoTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> Starcoder2Config:
        """
        Create a NeMo Starcoder2Config from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            Starcoder2Config: NeMo configuration for Starcoder2 models
        """
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
    """
    Exporter for converting NeMo Starcoder2Model to Hugging Face format.

    This class handles the conversion of NeMo's Starcoder2Model to Hugging Face's
    Starcoder2ForCausalLM format, including weight mapping and configuration translation.
    """

    def init(self) -> "Starcoder2ForCausalLM":
        """
        Initialize a HF Starcoder2ForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            Starcoder2ForCausalLM: Initialized HF Starcoder2 model
        """
        from transformers import Starcoder2ForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
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
        """
        Convert state dict from NeMo format to HF format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme.

        Args:
            source: Source NeMo model
            target: Target HF model

        Returns:
            The target model with weights transferred from source
        """
        mapping = {
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
                source_key="decoder.layers.*.self_attention.linear_qkv.bias",
                target_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                fn=TransformFns.split_qkv_bias,
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self):
        """
        Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFStarcoder2Config":
        """Create a HF HFStarcoder2Config from the NeMo model config.

        Translates the NeMo configuration parameters to the equivalent HF
        configuration.

        Returns:
            HFStarcoder2Config: HF configuration for Starcoder2 models
        """
        from transformers import Starcoder2Config as HFStarcoder2Config

        source: Starcoder2Config = io.load_context(str(self), subpath="model.config")

        return HFStarcoder2Config(
            architectures=["Starcoder2ForCausalLM"],
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


__all__ = [
    "Starcoder2Config",
    "Starcoder2Config3B",
    "Starcoder2Config7B",
    "Starcoder2Config15B",
    "Starcoder2Model",
]
