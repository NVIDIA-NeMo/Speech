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
from torch import nn

from nemo.collections.llm.fn.activation import squared_relu
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import NemotronConfig as HFNemotronConfig
    from transformers import NemotronForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class NemotronConfig(GPTConfig):
    """
    Configuration class for the Nemotron Config, inheriting from GPTConfig.
    """

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
    """
    Configuration class for the Nemotron3 4B Config, inheriting from NemotronConfig.
    """

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
    """
    Configuration class for the Nemotron3 8B Config, inheriting from NemotronConfig.
    """

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
    """
    Configuration class for the Nemotron3 22B Config, inheriting from NemotronConfig.
    """

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
    """
    Configuration class for the Nemotron4 15B Config, inheriting from NemotronConfig.
    """

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
    """
    Configuration class for the Nemotron4 340B Config, inheriting from NemotronConfig.
    """

    num_layers: int = 96
    seq_length: int = 4096
    hidden_size: int = 18432
    ffn_hidden_size: int = 73728
    num_attention_heads: int = 96
    num_query_groups: Optional[int] = 8
    kv_channels: Optional[int] = None
    init_method_std: float = 0.0063


class NemotronModel(GPTModel):
    """
    Nemotron model implementation based on the GPT model architecture.

    This class provides a high-level interface for Nemotron models,
    implementing the specific architecture and settings needed for Nemotron models.
    """

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
    """
    Importer for converting Hugging Face Nemotron models to NeMo format.

    This class handles the conversion of Hugging Face's NemotronForCausalLM models
    to NeMo's Nemotron format, including weight mapping and configuration translation.
    """

    def init(self) -> NemotronModel:
        """
        Initialize a NeMo NemotronModel instance.

        Returns:
            NemotronModel: Initialized NeMo Nemotron model with the appropriate configuration
                        and tokenizer.
        """
        return NemotronModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """
        Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
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
    def config(self) -> NemotronConfig:
        """
        Create a NeMo NemotronConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            NemotronConfig: NeMo configuration for Nemotron models
        """
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
            kv_channels=getattr(source, "head_dim", None),
        )

        return output


@io.model_exporter(NemotronModel, "hf")
class HFNemotronExporter(io.ModelConnector[NemotronModel, "NemotronForCausalLM"]):
    """
    Exporter for converting NeMo NemotronModel to Hugging Face format.

    This class handles the conversion of NeMo's NemotronModel to Hugging Face's
    NemotronForCausalLM format, including weight mapping and configuration translation.
    """

    def init(self, dtype=torch.bfloat16) -> "NemotronForCausalLM":
        """
        Initialize a HF NemotronForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            NemotronForCausalLM: Initialized HF Nemotron model
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
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
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp.up_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
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
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self):
        """
        Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFNemotronConfig":
        """Create a HF NemotronConfig from the NeMo model config.

        Translates the NeMo configuration parameters to the equivalent HF
        configuration.

        Returns:
            HFNemotronConfig: HF configuration for Nemotron models
        """
        from transformers import NemotronConfig as HFNemotronConfig

        source: NemotronConfig = io.load_context(str(self), subpath="model.config")

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


__all__ = [
    "NemotronConfig",
    "Nemotron3Config4B",
    "Nemotron3Config8B",
    "Nemotron3Config22B",
    "Nemotron4Config15B",
    "Nemotron4Config340B",
    "NemotronModel",
]
