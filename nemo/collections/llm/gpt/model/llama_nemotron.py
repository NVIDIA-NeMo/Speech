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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.gpt.model.llama import Llama31Config, Llama31Config8B, Llama31Config70B, LlamaConfig
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

if TYPE_CHECKING:
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class Llama31NemotronNano8BConfig(Llama31Config8B):
    """Configuration for an Llama31-Nemotron-Nano model."""

    kv_channels: int = 128


class Llama31Nemotron70BConfig(Llama31Config70B):
    """Configuration for an Llama31-Nemotron-70B model."""

    kv_channels: int = 128


class LlamaNemotronModel(GPTModel):
    """Llama-Nemotron model implementation based on the GPT model architecture.

    This class provides a high-level interface for Llama-Nemotron models,
    implementing the specific architecture and settings needed for Llama-Nemotron models.
    """

    def __init__(
        self,
        config: Annotated[Optional[LlamaConfig], Config[LlamaConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or Llama31NemotronNano8BConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(LlamaNemotronModel, "hf")
class HFLlamaNemotronImporter(io.ModelConnector["LlamaForCausalLM", LlamaNemotronModel]):
    """Importer for converting Hugging Face Llama-Nemotron models to NeMo format.

    This class handles the conversion of Hugging Face's LlamaForCausalLM models
    to NeMo's LlamaNemotronModel format, including weight mapping and configuration translation.
    """

    def init(self) -> LlamaNemotronModel:
        """Initialize a NeMo LlamaModel instance.

        Returns:
            LlamaModel: Initialized NeMo Llama model with the appropriate configuration
                        and tokenizer.
        """
        return LlamaNemotronModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import LlamaForCausalLM

        source = LlamaForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Llama-Nemotron model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Convert state dict from HF format to NeMo format.

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
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }
        if getattr(source.config, "tie_word_embeddings", False):
            del mapping["lm_head.weight"]

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
        """Get the tokenizer for the HF model.

        Returns:
            AutoTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> LlamaConfig:
        """Create a NeMo LlamaNemotronConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            LlamaConfig: NeMo configuration for Llama models
        """
        from transformers import GenerationConfig
        from transformers import LlamaConfig as HFLlamaConfig

        source = HFLlamaConfig.from_pretrained(str(self))
        generation_config = GenerationConfig.from_pretrained(str(self))
        print(generation_config)

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        assert getattr(source, 'rope_scaling', None), 'Llama-Nemotron model should have rope scaling'
        # Apply Llama3.1 customize rope scaling
        cls = partial(Llama31Config, scale_factor=source.rope_scaling.get("factor", 8.0))

        output = cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            kv_channels=getattr(source, "head_dim", None),
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            seq_length=source.max_position_embeddings,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
        )

        return output


@io.model_exporter(LlamaNemotronModel, "hf")
class HFLlamaNemotronExporter(io.ModelConnector[LlamaNemotronModel, "LlamaForCausalLM"]):
    """Exporter for converting NeMo Llama-Nemotron models to Hugging Face format.

    This class handles the conversion of NeMo's LlamaNemotronModel to Hugging Face's
    LlamaForCausalLM format, including weight mapping and configuration translation.
    """

    def init(self, dtype=torch.bfloat16) -> "LlamaForCausalLM":
        """Initialize a HF LlamaForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            LlamaForCausalLM: Initialized HF Llama model
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from NeMo to HF format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved HF model
        """
        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)

        target = target.cpu()
        if self.config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(output_path, state_dict=state_dict)
        else:
            target.save_pretrained(output_path)

        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target):
        """Convert state dict from NeMo format to HF format.

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
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        if not self.config.tie_word_embeddings:
            transforms.append(
                io.state_transform(
                    source_key="output_layer.weight",
                    target_key="lm_head.weight",
                    fn=TransformFns.prune_padding,
                )
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self) -> "TokenizerSpec":
        """Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self) -> "HFLlamaConfig":
        """Create a HF LlamaConfig from the NeMo model config.

        Translates the NeMo configuration parameters to the equivalent HF
        configuration.

        Returns:
            HFLlamaConfig: HF configuration for Llama models
        """
        source: LlamaConfig = io.load_context(str(self), subpath="model.config")

        from transformers import LlamaConfig as HFLlamaConfig

        rope_scaling = None
        # For Llama 3.1 and Llama 3.2, rope_scaling is used and thus needed to parsed to the config
        if isinstance(source, Llama31Config):
            rope_scaling = {
                'factor': source.scale_factor,
                'low_freq_factor': source.low_freq_factor,
                'high_freq_factor': source.high_freq_factor,
                'original_max_position_embeddings': source.old_context_len,
                'rope_type': 'llama3',
            }
        return HFLlamaConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=source.kv_channels,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            rope_scaling=rope_scaling,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
        )


__all__ = [
    "LlamaNemotronModel",
    "Llama31NemotronNano8BConfig",
]
