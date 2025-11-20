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
import json

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional, Union

import torch
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.gpt.model.llama import (
    Llama31Config,
    Llama31Config8B,
    Llama31Config70B,
    Llama31Config405B,
    LlamaConfig,
)
from nemo.collections.llm.gpt.model.llama_nemotron_config import (
    LLAMA_31_NEMOTRON_ULTRA_253B_HETEROGENEOUS_CONFIG,
    LLAMA_33_NEMOTRON_SUPER_49B_HETEROGENEOUS_CONFIG,
)
from nemo.collections.llm.utils import Config, is_safe_repo
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging
from nemo.utils.import_utils import safe_import

_, HAVE_TE = safe_import("transformer_engine")
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import get_gpt_heterogeneous_layer_spec
from megatron.core.transformer.heterogeneous.heterogeneous_config import HeterogeneousTransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

if TYPE_CHECKING:
    from peft import AutoPeftModelForCausalLM, PeftConfig
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


# Llama-Nemotron Super/Ultra uses heterogeneous architecture
def heterogeneous_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability.

    Uses Transformer Engine specs if available, otherwise falls back to local implementation.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: The selected module specification
    """
    return get_gpt_heterogeneous_layer_spec(config, use_te=HAVE_TE)


@dataclass
class Llama33NemotronSuper49BConfig(Llama31Config70B, HeterogeneousTransformerConfig):
    """Configuration for an Llama31-Nemotron-Super model."""

    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_layers: int = 80
    heterogeneous_layers_config_path: str = None
    heterogeneous_layers_config_encoded_json: str = LLAMA_33_NEMOTRON_SUPER_49B_HETEROGENEOUS_CONFIG
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = heterogeneous_layer_spec


@dataclass
class Llama31NemotronUltra253BConfig(Llama31Config405B, HeterogeneousTransformerConfig):
    """Configuration for an Llama31-Nemotron-Ultra model."""

    hidden_size: int = 16384
    num_attention_heads: int = 128
    num_layers: int = 162
    heterogeneous_layers_config_path: str = None
    heterogeneous_layers_config_encoded_json: str = LLAMA_31_NEMOTRON_ULTRA_253B_HETEROGENEOUS_CONFIG
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = heterogeneous_layer_spec


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

    def apply(self, output_path: Path, trust_remote_code: bool | None = None) -> Path:
        """Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved
            trust_remote_code: Whether remote code execution should be trusted for a given HF path

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import AutoModelForCausalLM, LlamaForCausalLM

        logging.info(f'Load HF model {str(self)}')
        self.trust_remote_code = trust_remote_code
        if 'Nano' in str(self):
            source = LlamaForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        else:
            source = AutoModelForCausalLM.from_pretrained(
                str(self),
                trust_remote_code=is_safe_repo(
                    trust_remote_code=self.trust_remote_code,
                    hf_path=str(self),
                ),
                torch_dtype='auto',
            )
        logging.info('Initialize NeMo Nemotron-Llama model')
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

        return AutoTokenizer(
            self.save_hf_tokenizer_assets(str(self)),
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=str(self),
            ),
        )

    @property
    def config(self) -> LlamaConfig:
        """Create a NeMo LlamaNemotronConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            LlamaConfig: NeMo configuration for Llama models
        """
        from transformers import AutoConfig, GenerationConfig

        source = AutoConfig.from_pretrained(
            str(self),
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=str(self),
            ),
        )
        try:
            generation_config = GenerationConfig.from_pretrained(str(self))
        except Exception:
            generation_config = None

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        assert getattr(source, 'rope_scaling', None), 'Llama-Nemotron model should have rope scaling'
        if getattr(source, 'block_configs', None) is not None:
            # Convert heterogeneous model (Llama-Nemotron Super/Ultra)
            target_class = (
                Llama33NemotronSuper49BConfig if source.num_hidden_layers == 80 else Llama31NemotronUltra253BConfig
            )
            cls = partial(
                target_class,
                heterogeneous_layers_config_encoded_json=source.to_json_string(),
                heterogeneous_layers_config_path=None,  # We directly load the block config as json
                scale_factor=source.rope_scaling.get("factor", 8.0),
                # For heterogeneous model, GQA is defined in each block config.
                # Llama-Nemotron has the same GQA across all non no-op attention layers.
                # We expose it to config.num_query_groups to make the merge_qkv work.
                # Here we assume block 0 is non no-ops for the attention
                num_query_groups=source.num_attention_heads // source.block_configs[0].attention.n_heads_in_group,
            )
        else:
            # Convert homogeneous model (Llama-Nemotron Nano/70B)
            target_class = Llama31NemotronNano8BConfig if source.num_hidden_layers == 32 else Llama31Nemotron70BConfig
            cls = partial(target_class, num_query_groups=source.num_key_value_heads)

        output = cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            kv_channels=getattr(source, "head_dim", None),
            scale_factor=source.rope_scaling.get('factor', 8.0),
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
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
    It supports both homogeneous (Nano/70B) and heterogeneous (Super/Ultra) model architectures.

    The exporter performs the following key operations:
    1. Initializes a Hugging Face model with appropriate configuration
    2. Maps weights from NeMo format to Hugging Face format
    3. Handles special cases for heterogeneous architectures
    4. Saves the converted model and tokenizer to the specified output path

    Attributes:
        tokenizer: The tokenizer associated with the NeMo model
        config: The configuration for the Hugging Face model

    Methods:
        init: Initialize a Hugging Face model instance
        apply: Convert and save the model to Hugging Face format
        convert_state: Convert model weights from NeMo to Hugging Face format
    """

    def init(self, dtype=torch.bfloat16, from_config=False, model_name=None) -> "LlamaForCausalLM":
        """Initialize a Hugging Face LlamaForCausalLM model instance.

        This method creates a new Hugging Face model instance with the appropriate configuration
        and data type. It handles both homogeneous and heterogeneous model architectures.

        Args:
            dtype (torch.dtype, optional): Data type for model parameters. Defaults to torch.bfloat16.
            from_config (bool, optional): Whether to initialize from config or load from pretrained.
                Set to True for homogeneous models (Nano/70B), False for heterogeneous models (Super/Ultra).
                Defaults to False.
            model_name (str, optional): Name of the pretrained model to load for heterogeneous architectures.
                Required when from_config is False. Defaults to None.

        Returns:
            LlamaForCausalLM: Initialized Hugging Face Llama model instance

        Raises:
            AssertionError: If model_name is not provided for heterogeneous models
        """
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            if from_config:
                # Llama-Nemotron Nano / Llama31Nemotron70BConfig
                return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

            # Llama-Nemotron Super/Ultra
            assert model_name is not None
            # Since Llama-Nemotron Super/Ultra is not importable from transformers, we can only initialize the HF model
            # from a known checkpoint folder containing the config file and modeling files.
            # The model_name will need to be passed in.
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=is_safe_repo(
                    trust_remote_code=self.trust_remote_code,
                    hf_path=model_name,
                ),
            )
            hf_model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=is_safe_repo(
                    trust_remote_code=self.trust_remote_code,
                    hf_path=model_name,
                ),
                torch_dtype=dtype,
            )
            # Register the AutoModel Hook so that the custom modeling files are saved during save_pretrained()
            type(hf_model).register_for_auto_class("AutoModelForCausalLM")
            return hf_model

    def apply(self, output_path: Path, target_model_name=None, trust_remote_code: bool | None = None) -> Path:
        """Convert and save a NeMo Llama-Nemotron model to Hugging Face format.

        This method performs the complete conversion process:
        1. Loads the NeMo model checkpoint
        2. Determines the appropriate target model configuration
        3. Initializes the Hugging Face model
        4. Converts and transfers the weights
        5. Saves the converted model and tokenizer

        Args:
            output_path (Path): Directory path where the converted model will be saved
            target_model_name (str, optional): Name of the target Hugging Face model.
                Required for heterogeneous models (Super/Ultra). For homogeneous models,
                this is determined automatically. Defaults to None.
            trust_remote_code: Whether remote code execution should be trusted for a given HF path

        Returns:
            Path: Path to the saved Hugging Face model directory

        Raises:
            ValueError: If the target model is not supported or if target_model_name is missing
                      for heterogeneous models
        """
        logging.info("Loading Llama-Nemotron NeMo checkpoint..")
        self.trust_remote_code = trust_remote_code
        source, _ = self.nemo_load(str(self))
        is_heterogeneous = isinstance(source.config, HeterogeneousTransformerConfig)
        if target_model_name is None:
            # Llama-Nemotron Super/Ultra uses custom modeling class
            if is_heterogeneous:
                num_layers = source.config.num_layers
                if num_layers == 80:
                    target_model_name = 'nvidia/Llama-3_3-Nemotron-Super-49B-v1'
                elif num_layers == 162:
                    target_model_name = 'nvidia/Llama-3_1-Nemotron-Ultra-253B-v1'
                else:
                    raise ValueError(
                        'Unknown target model. '
                        'Currently only support exporting Llama-Nemotron Nano/Super/Ultra models.'
                    )

        target = self.init(
            torch_dtype_from_mcore_config(source.config),
            from_config=not is_heterogeneous,
            model_name=target_model_name,
        )
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.tokenizer.save_pretrained(output_path)

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
    def tokenizer(self) -> "TokenizerSpec":
        """Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self) -> "HFLlamaConfig":
        """Create a HF LlamaConfig from the NeMo model config.
        This function should only be invoked for Non-heterogeneous transformers (i.e. Nano).

        Translates the NeMo configuration parameters to the equivalent HF
        configuration.

        Returns:
            HFLlamaConfig: HF configuration for Llama models
        """

        source: LlamaConfig = io.load_context(str(self), subpath="model.config")
        assert not isinstance(source, HeterogeneousTransformerConfig)

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


@io.model_exporter(LlamaNemotronModel, "hf-peft")
class HFLlamaNemotronPEFTExporter(HFLlamaNemotronExporter):
    """Exporter for converting NeMotron Llama models with PEFT adapters to Hugging Face format.

    This class extends HFLlamaNemotronExporter to handle Parameter-Efficient Fine-Tuning (PEFT)
    adapters, specifically LoRA and DoRA adapters.
    """

    def init(self, dtype=torch.bfloat16, from_config=False, model_name=None) -> "AutoPeftModelForCausalLM":
        """Initialize a HF PEFT model.

        Args:
            dtype: Data type for model parameters

        Returns:
            AutoPeftModelForCausalLM: Initialized HF PEFT model
        """
        from peft import get_peft_model

        model = super().init(dtype=dtype, from_config=from_config, model_name=model_name)

        # Infer base model checkpoint from checkpoint metadata file
        adapter_meta_path = ckpt_to_weights_subdir(str(self), is_saving=False) / ADAPTER_META_FILENAME
        with open(adapter_meta_path, "r") as f:
            model_ckpt_path = json.load(f)['model_ckpt_path']
        model.name_or_path = '/'.join(model_ckpt_path.split("/")[-2:])

        return get_peft_model(model, self.peft_config, autocast_adapter_dtype=False)

    def apply(self, output_path: Path, target_model_name=None) -> Path:
        """Apply the conversion from NeMo PEFT model to HF format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved HF PEFT model
        """
        from nemo.collections.llm.peft import CanonicalLoRA, DoRA, LoRA

        self.peft_obj: Union[LoRA, DoRA, CanonicalLoRA] = io.load_context(str(self), subpath="model.model_transform")

        source, _ = self.nemo_load(str(self))
        is_heterogeneous = isinstance(source.config, HeterogeneousTransformerConfig)
        if target_model_name is None:
            # Llama-Nemotron Super/Ultra uses customize modeling class
            if is_heterogeneous:
                num_layers = source.config.num_layers
                if num_layers == 80:
                    target_model_name = 'nvidia/Llama-3_3-Nemotron-Super-49B-v1'
                elif num_layers == 162:
                    target_model_name = 'nvidia/Llama-3_1-Nemotron-Ultra-253B-v1'
                else:
                    raise ValueError(
                        'Unknown target model. '
                        'Currently only support exporting Llama-Nemotron Nano/Super/Ultra models.'
                    )

        target = self.init(
            torch_dtype_from_mcore_config(source.config),
            from_config=not is_heterogeneous,
            model_name=target_model_name,
        )
        target = self.convert_state(source, target)
        target = target.cpu()
        target.save_pretrained(output_path, save_embedding_layers=False)

        return output_path

    def convert_state(self, source, target):
        """Convert state dict from NeMo PEFT model to HF PEFT format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme for PEFT adapters.

        Args:
            source: Source NeMo model with PEFT adapters
            target: Target HF model

        Returns:
            The target model with weights transferred from source
        """
        from nemo.collections.llm.peft import CanonicalLoRA

        # nemo and HF prefixes
        pn = "decoder.layers."
        ph = "base_model.model.model.layers."

        # linear_proj and linear_fc2 prefixes
        p_proj = "self_attention.linear_proj.adapter"
        p_fc2 = "mlp.linear_fc2.adapter"

        # linear_qkv and linear_fc1 prefixes
        p_qkv = "self_attention.linear_qkv.adapter"
        p_fc1 = "mlp.linear_fc1.adapter"

        mapping = {
            # linear_proj for both canonical and performant lora
            f"{pn}*.{p_proj}.linear_in.weight": f"{ph}*.self_attn.o_proj.lora_A.default.weight",
            f"{pn}*.{p_proj}.linear_out.weight": f"{ph}*.self_attn.o_proj.lora_B.default.weight",
            # linear_fc2 for both canonical and performant lora
            f"{pn}*.{p_fc2}.linear_in.weight": f"{ph}*.mlp.down_proj.lora_A.default.weight",
            f"{pn}*.{p_fc2}.linear_out.weight": f"{ph}*.mlp.down_proj.lora_B.default.weight",
        }
        transforms = []

        if isinstance(self.peft_obj, CanonicalLoRA):
            mapping.update(
                {
                    # linear_qkv for canonical lora
                    f"{pn}*.{p_qkv}.adapter_q.linear_in.weight": f"{ph}*.self_attn.q_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_q.linear_out.weight": f"{ph}*.self_attn.q_proj.lora_B.default.weight",
                    f"{pn}*.{p_qkv}.adapter_k.linear_in.weight": f"{ph}*.self_attn.k_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_k.linear_out.weight": f"{ph}*.self_attn.k_proj.lora_B.default.weight",
                    f"{pn}*.{p_qkv}.adapter_v.linear_in.weight": f"{ph}*.self_attn.v_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_v.linear_out.weight": f"{ph}*.self_attn.v_proj.lora_B.default.weight",
                    # linear_fc1 for canonical lora
                    f"{pn}*.{p_fc1}.adapter_up.linear_in.weight": f"{ph}*.mlp.up_proj.lora_A.default.weight",
                    f"{pn}*.{p_fc1}.adapter_up.linear_out.weight": f"{ph}*.mlp.up_proj.lora_B.default.weight",
                    f"{pn}*.{p_fc1}.adapter_gate.linear_in.weight": f"{ph}*.mlp.gate_proj.lora_A.default.weight",
                    f"{pn}*.{p_fc1}.adapter_gate.linear_out.weight": f"{ph}*.mlp.gate_proj.lora_B.default.weight",
                }
            )
        else:
            transforms.extend(
                [
                    # linear_qkv for performant lora
                    io.state_transform(
                        source_key=f"{pn}*.self_attention.linear_qkv.adapter.linear_in.weight",
                        target_key=(
                            f"{ph}*.self_attn.q_proj.lora_A.default.weight",
                            f"{ph}*.self_attn.k_proj.lora_A.default.weight",
                            f"{ph}*.self_attn.v_proj.lora_A.default.weight",
                        ),
                        fn=TransformFns.duplicate3,
                    ),
                    io.state_transform(
                        source_key=f"{pn}*.self_attention.linear_qkv.adapter.linear_out.weight",
                        target_key=(
                            f"{ph}*.self_attn.q_proj.lora_B.default.weight",
                            f"{ph}*.self_attn.k_proj.lora_B.default.weight",
                            f"{ph}*.self_attn.v_proj.lora_B.default.weight",
                        ),
                        fn=TransformFns.split_qkv,
                    ),
                    # linear_fc1 for performant lora
                    io.state_transform(
                        source_key=f"{pn}*.mlp.linear_fc1.adapter.linear_in.weight",
                        target_key=(
                            f"{ph}*.mlp.gate_proj.lora_A.default.weight",
                            f"{ph}*.mlp.up_proj.lora_A.default.weight",
                        ),
                        fn=TransformFns.duplicate2,
                    ),
                    io.state_transform(
                        source_key=f"{pn}*.mlp.linear_fc1.adapter.linear_out.weight",
                        target_key=(
                            f"{ph}*.mlp.gate_proj.lora_B.default.weight",
                            f"{ph}*.mlp.up_proj.lora_B.default.weight",
                        ),
                        fn=TransformFns.split_fc1,
                    ),
                ]
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def peft_config(self) -> "PeftConfig":
        """Create a PEFT config for the HF model.

        Translates the NeMo PEFT configuration to the equivalent HF PEFT
        configuration.

        Returns:
            PeftConfig: HF PEFT configuration
        """
        from peft import LoraConfig

        from nemo.collections.llm.peft import DoRA

        assert (
            not self.peft_obj.dropout or self.peft_obj.dropout_position == 'pre'
        ), "LoRA dropout_position must be 'pre' to convert to HF."

        NEMO2HF = {
            'linear_q': ['q_proj'],
            'linear_k': ['k_proj'],
            'linear_v': ['v_proj'],
            'linear_qkv': ['q_proj', 'k_proj', 'v_proj'],
            'linear_proj': ['o_proj'],
            'linear_fc1_up': ['up_proj'],
            'linear_fc1_gate': ['gate_proj'],
            'linear_fc1': ['up_proj', 'gate_proj'],
            'linear_fc2': ['down_proj'],
        }

        # Infer HF target modules from NeMo target modules
        hf_target_modules = []
        for tm in self.peft_obj.target_modules:
            hf_target_modules.extend(NEMO2HF[tm])

        return LoraConfig(
            r=self.peft_obj.dim,
            target_modules=hf_target_modules,
            lora_alpha=self.peft_obj.alpha,
            lora_dropout=self.peft_obj.dropout,
            use_dora=isinstance(self.peft_obj, DoRA),
        )


__all__ = [
    "LlamaNemotronModel",
    "Llama31NemotronNano8BConfig",
    "Llama33NemotronSuper49BConfig",
    "Llama31NemotronUltra253BConfig",
    "Llama31Nemotron70BConfig",
]
