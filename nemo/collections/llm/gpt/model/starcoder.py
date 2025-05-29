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
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import GPTBigCodeConfig as HFStarcoderConfig
    from transformers import GPTBigCodeForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class StarcoderConfig(GPTConfig):
    """
    Configuration class for the Starcoder Config, inheriting from GPTConfig.
    """

    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    activation_func: Callable = F.gelu
    add_bias_linear: bool = True
    seq_length: int = 8192
    position_embedding_type: str = "learned_absolute"
    hidden_dropout: float = 0.2
    attention_dropout: float = 0.2
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-5
    share_embeddings_and_output_weights: bool = False
    kv_channels: int = None
    num_query_groups: int = 1
    attention_softmax_in_fp32: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class StarcoderConfig15B(StarcoderConfig):
    """
    Configuration class for the Starcoder 15B Config, inheriting from StarcoderConfig.
    """

    num_layers: int = 40
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    init_method_std: float = 0.02


class StarcoderModel(GPTModel):
    """
    Starcoder model implementation based on the GPT model architecture.

    This class provides a high-level interface for Starcoder models,
    implementing the specific architecture and settings needed for Starcoder models.
    """

    def __init__(
        self,
        config: Annotated[Optional[StarcoderConfig], Config[StarcoderConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or StarcoderConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(StarcoderModel, "hf")
class HFStarcoderImporter(io.ModelConnector["GPTBigCodeForCausalLM", StarcoderModel]):
    """
    Importer for converting Hugging Face Starcoder models to NeMo format.

    This class handles the conversion of Hugging Face's GPTBigCodeForCausalLM models
    to NeMo's Starcoder format, including weight mapping and configuration translation.
    """

    def init(self) -> StarcoderModel:
        """
        Initialize a NeMo StarcoderModel instance.

        Returns:
            StarcoderModel: Initialized NeMo Starcoder model with the appropriate configuration
                        and tokenizer.
        """
        return StarcoderModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """
        Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import GPTBigCodeForCausalLM

        source = GPTBigCodeForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Starcoder model to Nemo, model saved to {output_path}")

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
            "transformer.wte.weight": "embedding.word_embeddings.weight",
            "transformer.wpe.weight": "embedding.position_embeddings.weight",
            "transformer.h.*.attn.c_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "transformer.h.*.attn.c_proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            "transformer.h.*.attn.c_attn.weight": "decoder.layers.*.self_attention.linear_qkv.weight",
            "transformer.h.*.attn.c_attn.bias": "decoder.layers.*.self_attention.linear_qkv.bias",
            "transformer.h.*.mlp.c_fc.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "transformer.h.*.mlp.c_fc.bias": "decoder.layers.*.mlp.linear_fc1.bias",
            "transformer.h.*.mlp.c_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "transformer.h.*.mlp.c_proj.bias": "decoder.layers.*.mlp.linear_fc2.bias",
            "transformer.h.*.ln_1.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "transformer.h.*.ln_1.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "transformer.h.*.ln_2.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "transformer.h.*.ln_2.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "transformer.ln_f.weight": "decoder.final_layernorm.weight",
            "transformer.ln_f.bias": "decoder.final_layernorm.bias",
            "lm_head.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping)

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
    def config(self) -> StarcoderConfig:
        """
        Create a NeMo StarcoderConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            StarcoderConfig: NeMo configuration for Starcoder models
        """
        from transformers import GPTBigCodeConfig as HFStarcoderConfig

        source = HFStarcoderConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = StarcoderConfig(
            num_layers=source.n_layer,
            hidden_size=source.n_embd,
            ffn_hidden_size=source.n_inner,
            num_attention_heads=source.n_head,
            init_method_std=source.initializer_range,
            seq_length=source.n_positions,
            layernorm_epsilon=source.layer_norm_epsilon,
            num_query_groups=1,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(StarcoderModel, "hf")
class HFStarcoderExporter(io.ModelConnector[StarcoderModel, "GPTBigCodeForCausalLM"]):
    """
    Exporter for converting NeMo StarcoderModel to Hugging Face format.

    This class handles the conversion of NeMo's StarcoderModel to Hugging Face's
    GPTBigCodeForCausalLM format, including weight mapping and configuration translation.
    """

    def init(self, dtype=torch.bfloat16) -> "GPTBigCodeForCausalLM":
        """
        Initialize a HF GPTBigCodeForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            GPTBigCodeForCausalLM: Initialized HF Starcoder model
        """
        from transformers import GPTBigCodeForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return GPTBigCodeForCausalLM._from_config(self.config, torch_dtype=dtype)

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
            "embedding.position_embeddings.weight": "transformer.wpe.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "transformer.h.*.attn.c_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.bias": "transformer.h.*.attn.c_proj.bias",
            "decoder.layers.*.self_attention.linear_qkv.weight": "transformer.h.*.attn.c_attn.weight",
            "decoder.layers.*.self_attention.linear_qkv.bias": "transformer.h.*.attn.c_attn.bias",
            "decoder.layers.*.mlp.linear_fc1.weight": "transformer.h.*.mlp.c_fc.weight",
            "decoder.layers.*.mlp.linear_fc1.bias": "transformer.h.*.mlp.c_fc.bias",
            "decoder.layers.*.mlp.linear_fc2.weight": "transformer.h.*.mlp.c_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.bias": "transformer.h.*.mlp.c_proj.bias",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "transformer.h.*.ln_1.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "transformer.h.*.ln_1.bias",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "transformer.h.*.ln_2.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "transformer.h.*.ln_2.bias",
            "decoder.final_layernorm.weight": "transformer.ln_f.weight",
            "decoder.final_layernorm.bias": "transformer.ln_f.bias",
        }

        transforms = [
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="transformer.wte.weight",
                fn=TransformFns.prune_padding,
            ),
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
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
    def config(self) -> "HFStarcoderConfig":
        """Create a HF GPTBigCodeConfig from the NeMo model config.

        Translates the NeMo configuration parameters to the equivalent HF
        configuration.

        Returns:
            HFStarcoderConfig: HF configuration for Starcoder models
        """
        from transformers import GPTBigCodeConfig as HFStarcoderConfig

        source: StarcoderConfig = io.load_context(str(self), subpath="model.config")

        return HFStarcoderConfig(
            architectures=["GPTBigCodeForCausalLM"],
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
            vocab_size=self.tokenizer.vocab_size,
        )


__all__ = [
    "StarcoderConfig",
    "StarcoderConfig15B",
    "StarcoderModel",
]
