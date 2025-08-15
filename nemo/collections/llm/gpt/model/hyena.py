# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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


import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Type

import torch
from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer_specs import hyena_stack_spec, hyena_stack_spec_no_te
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_model import HyenaModel as MCoreHyenaModel
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import hyena_no_weight_decay_cond
from nemo.lightning import get_vocab_size, io, teardown
from nemo.lightning.base import NEMO_MODELS_CACHE
from nemo.lightning.io.state import TransformFns
from nemo.utils import logging


class HyenaInferenceContext(StaticInferenceContext):
    """Hyena-specific inference context."""

    def reset(self):
        """Reset the inference context."""
        super().reset()  # standard state reset for GPT models
        for key in dir(self):
            # Remove all of the state that we add in hyena.py
            if "filter_state_dict" in key:
                delattr(self, key)


class HyenaModel(GPTModel):
    """
    This is a wrapper around the MCoreHyenaModel to allow for inference. Our model follows the same API
      as the GPTModel, but the megatron model class is different so we need to handle the inference wrapper
      slightly differently.
    """

    def get_inference_wrapper(
        self, params_dtype, inference_batch_times_seqlen_threshold, inference_max_seq_length=None
    ) -> torch.Tensor:
        """
        Gets the inference wrapper for the Hyena model.

        Args:
            params_dtype: The data type for model parameters
            inference_batch_times_seqlen_threshold: Threshold for batch size * sequence length during inference
            inference_max_seq_length: Maximum sequence length for inference

        Returns:
            GPTInferenceWrapper: The inference wrapper for the model

        Raises:
            ValueError: If MCoreHyenaModel instance not found or vocab size cannot be determined
        """
        # This is to get the MCore model required in GPTInferenceWrapper.
        mcore_model = self.module
        while mcore_model:
            if type(mcore_model) is MCoreHyenaModel:
                break
            mcore_model = getattr(mcore_model, "module", None)
        if mcore_model is None or type(mcore_model) is not MCoreHyenaModel:
            raise ValueError("Exact MCoreHyenaModel instance not found in the model structure.")

        vocab_size = None
        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.config, 'vocab_size'):
            vocab_size = self.config.vocab_size
        else:
            raise ValueError(
                'Unable to find vocab size.'
                ' Either pass in a tokenizer with vocab size, or set vocab size in the model config'
            )

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
            inference_max_seq_length=inference_max_seq_length,
            inference_max_requests=1,
        )

        inference_context = HyenaInferenceContext.from_config(inference_wrapper_config)
        model_inference_wrapper = GPTInferenceWrapper(mcore_model, inference_wrapper_config, inference_context)
        return model_inference_wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """
        Forward pass of the Hyena model.

        Args:
            input_ids: Input token IDs
            position_ids: Position IDs for input tokens
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            decoder_input: Optional decoder input
            loss_mask: Optional loss mask
            inference_context: Optional inference parameters
            packed_seq_params: Optional parameters for packed sequences

        Returns:
            torch.Tensor: Output tensor from the model
        """
        extra_kwargs = {'packed_seq_params': packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            loss_mask=loss_mask,
            **extra_kwargs,
        )
        return output_tensor


def hyena_forward_step(model, batch) -> torch.Tensor:
    """
    Performs a forward step for the Hyena model.

    Args:
        model: The Hyena model
        batch: Dictionary containing input batch data with keys:
            - tokens: Input token IDs
            - position_ids: Position IDs
            - labels: Labels for loss computation
            - loss_mask: Mask for loss computation

    Returns:
        torch.Tensor: Output from the model forward pass
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
        "loss_mask": batch["loss_mask"],
    }
    forward_args["attention_mask"] = None
    return model(**forward_args)


@dataclass
class HyenaConfig(TransformerConfig, io.IOMixin):
    """
    Configuration dataclass for Hyena.

    For adjusting ROPE when doing context extension, set seq_len_interpolation_factor relative to 8192.
    For example, if your context length is 512k, then set the factor to 512k / 8k = 64.
    """

    # From megatron.core.models.hyena.hyena_model.HyenaModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    num_layers: int = 2
    hidden_size: int = 1024
    num_attention_heads: int = 8
    num_groups_hyena: int = None
    num_groups_hyena_medium: int = None
    num_groups_hyena_short: int = None
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str = None
    post_process: bool = True
    pre_process: bool = True
    seq_length: int = 2048
    position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'rope'
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = True
    fp32_residual_connection: bool = True
    normalization: str = 'RMSNorm'
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    attention_backend: AttnBackend = AttnBackend.flash
    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 4
    forward_step_fn: Callable = hyena_forward_step
    data_step_fn: Callable = gpt_data_step
    tokenizer_model_path: str = None
    hyena_init_method: str = None
    hyena_output_layer_init_method: str = None
    hyena_filter_no_wd: bool = True
    remove_activation_post_first_layer: bool = True
    add_attn_proj_bias: bool = True
    cross_entropy_loss_fusion: bool = False  # Faster but lets default to False for more precision
    tp_comm_overlap: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    add_bias_output: bool = False
    use_te: bool = True
    to_upper: str = "normalized_weighted"  # choose between "weighted" and "normalized_weighted"
    use_short_conv_bias: bool = False
    # Use this if you want to turn FP8 on for the linear layer in the mixer only. When using this, do not set
    #  Fp8 in the mixed precision plugin.
    vortex_style_fp8: bool = False
    use_subquadratic_ops: bool = False
    share_embeddings_and_output_weights: bool = True

    def __post_init__(self):
        """
        Post-initialization hook that sets up weight decay conditions.
        """
        super().__post_init__()
        self.hyena_no_weight_decay_cond_fn = hyena_no_weight_decay_cond if self.hyena_filter_no_wd else None

    def configure_model(self, tokenizer, vp_stage: Optional[int] = None) -> "MCoreHyenaModel":
        """
        Configures and returns a Hyena model instance based on the config settings.

        Args:
            tokenizer: Tokenizer to use for the model
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreHyenaModel: Configured Hyena model instance
        """
        self.bias_activation_fusion = False if self.remove_activation_post_first_layer else self.bias_activation_fusion

        assert (
            getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None
        ), "Virtual pipeline model parallelism is temporarily unsupported in Hyena."

        model = MCoreHyenaModel(
            self,
            hyena_stack_spec=hyena_stack_spec if self.use_te else hyena_stack_spec_no_te,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            num_groups_hyena=self.num_groups_hyena,
            num_groups_hyena_medium=self.num_groups_hyena_medium,
            num_groups_hyena_short=self.num_groups_hyena_short,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            hyena_init_method=self.hyena_init_method,
            hyena_output_layer_init_method=self.hyena_output_layer_init_method,
            remove_activation_post_first_layer=self.remove_activation_post_first_layer,
            add_attn_proj_bias=self.add_attn_proj_bias,
        )
        return model


@dataclass
class HyenaTestConfig(HyenaConfig):
    """Configuration for testing Hyena models."""

    hybrid_override_pattern: str = "SDH*"
    num_layers: int = 4
    seq_length: int = 8192
    hidden_size: int = 4096
    num_groups_hyena: int = 4096
    num_groups_hyena_medium: int = 256
    num_groups_hyena_short: int = 256
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 11008
    gated_linear_unit: bool = True
    num_attention_heads: int = 32
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 2
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True
    use_short_conv_bias: bool = False
    use_subquadratic_ops: bool = False


@dataclass
class HyenaNVTestConfig(HyenaTestConfig):
    """
    Several unintentional design choices were made to the original Arc implementation that are required to use the
    original Arc checkpoints, but may result in less stable model training. If you are training from scratch,
    these are the recommended configs.
    """

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False
    use_short_conv_bias: bool = True


@dataclass
class Hyena1bConfig(HyenaConfig):
    """Config matching the 1b 8k context Evo2 model"""

    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*"
    num_layers: int = 25
    recompute_num_layers: int = 5  # needs to be a multiple of num_layers
    seq_length: int = 8192
    hidden_size: int = 1920
    num_groups_hyena: int = 1920
    num_groups_hyena_medium: int = 128
    num_groups_hyena_short: int = 128
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 5120
    gated_linear_unit: bool = True
    num_attention_heads: int = 15
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 5
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True


@dataclass
class HyenaNV1bConfig(Hyena1bConfig):
    """
    Several unintentional design choices were made to the original Arc implementation that are required to use the
    original Arc checkpoints, but may result in less stable model training. If you are training from scratch,
    these are the recommended configs.
    """

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False
    use_short_conv_bias: bool = True


@dataclass
class Hyena7bConfig(HyenaConfig):
    """Config matching the 7b 8k context Evo2 model"""

    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*"
    num_layers: int = 32
    seq_length: int = 8192
    hidden_size: int = 4096
    num_groups_hyena: int = 4096
    num_groups_hyena_medium: int = 256
    num_groups_hyena_short: int = 256
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 11008
    gated_linear_unit: bool = True
    num_attention_heads: int = 32
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 4
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True


@dataclass
class HyenaNV7bConfig(Hyena7bConfig):
    """
    Several unintentional design choices were made to the original Arc implementation that are required to use the
    original Arc checkpoints, but may result in less stable model training. If you are training from scratch,
    these are the recommended configs.
    """

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False
    use_short_conv_bias: bool = True


@dataclass
class Hyena40bConfig(HyenaConfig):
    """Config matching the 40b 8k context Evo2 model"""

    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*SDH*SDHSDH*SDHSDH*"
    num_layers: int = 50
    seq_length: int = 8192
    hidden_size: int = 8192
    num_groups_hyena: int = 8192
    num_groups_hyena_medium: int = 512
    num_groups_hyena_short: int = 512
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 21888
    gated_linear_unit: bool = True
    num_attention_heads: int = 64
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 2
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True


@dataclass
class HyenaNV40bConfig(Hyena40bConfig):
    """
    Several unintentional design choices were made to the original Arc implementation that are required to use the
    original Arc checkpoints, but may result in less stable model training. If you are training from scratch,
    these are the recommended configs.
    """

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False
    use_short_conv_bias: bool = True


@dataclass
class Hyena7bARCLongContextConfig(Hyena7bConfig):
    """The checkpoint from ARC requires padding to the FFN dim
    due to constraintes from large TP size for training."""

    ffn_hidden_size: int = 11264
    seq_len_interpolation_factor: float = 128


@dataclass
class Hyena40bARCLongContextConfig(Hyena40bConfig):
    """The checkpoint from ARC requires padding to the FFN dim
    due to constraintes from large TP size for training."""

    ffn_hidden_size: int = 22528
    seq_len_interpolation_factor: float = 128


@io.model_importer(HyenaModel, "pytorch")
class PyTorchHyenaImporter(io.ModelConnector["HyenaModel", HyenaModel]):
    """
    Importer class for converting PyTorch Hyena models to NeMo format.
    """

    def __new__(cls, path: str, model_config=None):
        """
        Creates a new importer instance.

        Args:
            path: Path to the PyTorch model
            model_config: Optional model configuration

        Returns:
            PyTorchHyenaImporter instance
        """
        instance = super().__new__(cls, path)
        instance.model_config = model_config
        return instance

    def init(self) -> HyenaModel:
        """
        Initializes a new HyenaModel instance.

        Returns:
            HyenaModel: Initialized model
        """
        return HyenaModel(self.config, tokenizer=self.tokenizer)

    def get_source_model(self):
        """
        Returns the source model.
        """
        return torch.load(str(self), map_location='cpu')

    def apply(self, output_path: Path, checkpoint_format: str = 'torch_dist') -> Path:
        """
        Applies the model conversion from PyTorch to NeMo format.

        Args:
            output_path: Path to save the converted model
            checkpoint_format: Format for saving checkpoints

        Returns:
            Path: Path to the saved NeMo model
        """
        source = self.get_source_model()

        if 'model' in source:
            source = source['model']

        class ModelState:
            """Wrapper around the source model state dictionary that also handles some weight transformations."""

            def __init__(self, state_dict, num_layers, fp32_suffixes):
                """Wrapper around the source model state dictionary that also handles some weight transformations.

                Args:
                    state_dict: original state dictionary from the source model
                    num_layers: number of layers in the source model
                """
                self.num_layers = num_layers
                state_dict = self.transform_source_dict(state_dict)
                self._state_dict = state_dict
                self.fp32_suffixes = fp32_suffixes

            def state_dict(self):
                """Return the state dictionary."""
                return self._state_dict

            def to(self, dtype):
                """Convert the state dictionary to the target dtype."""
                for k, v in self._state_dict.items():
                    if "_extra" not in k:
                        if v.dtype != dtype:
                            logging.warning(f"Converting {k} from {v.dtype} (source model) to {dtype} (target model)")
                        k_suffix = k.split('.')[-1]
                        if k_suffix in self.fp32_suffixes:
                            _dtype = torch.float32
                        else:
                            _dtype = dtype
                        self._state_dict[k] = v.to(_dtype)

            def adjust_medium_filter(self, updated_data):
                """Adjust the medium filter."""
                from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig

                for k, v in updated_data.items():
                    if "filter.h" in k or "filter.decay" in k:
                        updated_data[k] = v[:, : HyenaConfig().hyena_medium_conv_len]
                return updated_data

            def transform_source_dict(self, source):
                """Transform the source state dictionary, applying some challenging layer name re-mappings and
                removing extra keys, as well as truncating a filter that didn't need to extend to the full
                sequence length dim.
                """
                import re

                layer_map = {i + 2: i for i in range(self.num_layers)}
                layer_map[self.num_layers + 3] = self.num_layers
                updated_data = {}

                for key in list(source['module'].keys()):
                    if "_extra" in key:
                        source['module'].pop(key)
                    else:
                        match = re.search(r'sequential\.(\d+)', key)
                        if match:
                            original_layer_num = int(match.group(1))
                            if original_layer_num in layer_map:
                                # Create the updated key by replacing the layer number
                                new_key = re.sub(rf'\b{original_layer_num}\b', str(layer_map[original_layer_num]), key)
                                updated_data[new_key] = source['module'][key]
                            else:
                                # Keep the key unchanged if no mapping exists
                                updated_data[key] = source['module'][key]
                        else:
                            updated_data[key] = source['module'][key]
                updated_data = self.adjust_medium_filter(updated_data)
                return updated_data

        target = self.init()
        trainer = self.nemo_setup(target, ckpt_async_save=False, save_ckpt_format=checkpoint_format)
        target.to(self.config.params_dtype)
        fp32_suffixes = {n.split('.')[-1] for n, p in target.named_parameters() if p.dtype == torch.float32}
        source = ModelState(source, self.config.num_layers, fp32_suffixes)
        source.to(self.config.params_dtype)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted Hyena model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """
        Converts the state dictionary from source format to target format.

        Args:
            source: Source model state
            target: Target model

        Returns:
            Result of applying state transforms
        """
        mapping = {}
        mapping['sequential.0.word_embeddings.weight'] = 'embedding.word_embeddings.weight'
        mapping[f'sequential.{len(self.config.hybrid_override_pattern)}.norm.weight'] = 'decoder.final_norm.weight'
        te_enabled = self.config.use_te
        for i, symbol in enumerate(self.config.hybrid_override_pattern):
            if te_enabled:
                mapping[f'sequential.{i}.pre_mlp_layernorm.weight'] = (
                    f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'
                )
            else:
                mapping[f'sequential.{i}.pre_mlp_layernorm.weight'] = f'decoder.layers.{i}.pre_mlp_layernorm.weight'
            mapping[f'sequential.{i}.mlp.w3.weight'] = f'decoder.layers.{i}.mlp.linear_fc2.weight'

            if symbol != '*':
                if te_enabled:
                    mapping[f'sequential.{i}.input_layernorm.weight'] = (
                        f'decoder.layers.{i}.mixer.dense_projection.layer_norm_weight'
                    )
                else:
                    mapping[f'sequential.{i}.input_layernorm.weight'] = f'decoder.layers.{i}.norm.weight'

                mapping[f'sequential.{i}.mixer.dense_projection.weight'] = (
                    f'decoder.layers.{i}.mixer.dense_projection.weight'
                )
                mapping[f'sequential.{i}.mixer.hyena_proj_conv.short_conv_weight'] = (
                    f'decoder.layers.{i}.mixer.hyena_proj_conv.short_conv_weight'
                )
                mapping[f'sequential.{i}.mixer.dense.weight'] = f'decoder.layers.{i}.mixer.dense.weight'
                mapping[f'sequential.{i}.mixer.dense.bias'] = f'decoder.layers.{i}.mixer.dense.bias'

                if symbol == 'S':
                    mapping[f'sequential.{i}.mixer.mixer.short_conv.short_conv_weight'] = (
                        f'decoder.layers.{i}.mixer.mixer.short_conv.short_conv_weight'
                    )

                elif symbol == 'D':
                    mapping[f'sequential.{i}.mixer.mixer.conv_bias'] = f'decoder.layers.{i}.mixer.mixer.conv_bias'
                    mapping[f'sequential.{i}.mixer.mixer.filter.h'] = f'decoder.layers.{i}.mixer.mixer.filter.h'
                    mapping[f'sequential.{i}.mixer.mixer.filter.decay'] = (
                        f'decoder.layers.{i}.mixer.mixer.filter.decay'
                    )

                elif symbol == 'H':
                    mapping[f'sequential.{i}.mixer.mixer.conv_bias'] = f'decoder.layers.{i}.mixer.mixer.conv_bias'
                    mapping[f'sequential.{i}.mixer.mixer.filter.gamma'] = (
                        f'decoder.layers.{i}.mixer.mixer.filter.gamma'
                    )
                    mapping[f'sequential.{i}.mixer.mixer.filter.R'] = f'decoder.layers.{i}.mixer.mixer.filter.R'
                    mapping[f'sequential.{i}.mixer.mixer.filter.p'] = f'decoder.layers.{i}.mixer.mixer.filter.p'

            elif symbol == '*':
                if te_enabled:
                    mapping[f'sequential.{i}.input_layernorm.weight'] = (
                        f'decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight'
                    )
                else:
                    mapping[f'sequential.{i}.input_layernorm.weight'] = f'decoder.layers.{i}.input_layernorm.weight'

                mapping[f'sequential.{i}.mixer.dense_projection.weight'] = (
                    f'decoder.layers.{i}.self_attention.linear_qkv.weight'
                )
                mapping[f'sequential.{i}.mixer.dense.weight'] = f'decoder.layers.{i}.self_attention.linear_proj.weight'
                mapping[f'sequential.{i}.mixer.dense.bias'] = f'decoder.layers.{i}.self_attention.linear_proj.bias'
            else:
                raise ValueError(f'Unknown symbol: {symbol}')

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                # Transforms that are more complicated than a simple mapping of an old key name to a new one:
                io.state_transform(
                    source_key=("sequential.*.mlp.w1.weight", "sequential.*.mlp.w2.weight"),
                    target_key="decoder.layers.*.mlp.linear_fc1.weight",
                    fn=TransformFns.merge_fc1,
                )
            ],
        )

    @property
    def tokenizer(self):
        """
        Gets the tokenizer for the model.

        Returns:
            Tokenizer instance
        """
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        tokenizer = get_nmt_tokenizer(
            library=self.model_config.tokenizer_library,
        )

        return tokenizer

    @property
    def config(self) -> HyenaConfig:
        """
        Gets the model configuration.

        Returns:
            HyenaConfig: Model configuration
        """
        return self.model_config


@io.model_importer(HyenaModel, "hf")
class HuggingFaceSavannaHyenaImporter(PyTorchHyenaImporter):
    """
    Importer class for converting HuggingFace Savanna Hyena models to NeMo format.
        See: https://huggingface.co/arcinstitute/savanna_evo2_7b for an example of a savanna model that this can
        import and convert to NeMo format. Any of the Arc models that start with "savanna_" should work.
    """

    def get_source_model(self):
        """
        Returns the source model.
        """
        import huggingface_hub.errors
        from huggingface_hub import hf_hub_download

        if os.path.exists(str(self)):
            logging.info(f"Loading model from local path {str(self)}")
            return torch.load(str(self), map_location='cpu', weights_only=False)
        else:
            if ":" in str(self):
                repo_id, revision = str(self).split(":")
            else:
                repo_id = str(self)
                revision = None
            # See HF download logic here:
            #   https://github.com/ArcInstitute/evo2/blob/96ac9d9cd/evo2/models.py#L191-L231
            modelname = repo_id.split("/")[-1]
            download_dir = str(NEMO_MODELS_CACHE / repo_id)
            weights_filename = f"{modelname}.pt"
            try:
                weights_path = hf_hub_download(
                    repo_id=repo_id, local_dir=download_dir, revision=revision, filename=weights_filename
                )
            except Exception:
                # Try downloading multi-part
                # If file is split, download and join parts
                logging.warning(f"Single path download failed, try loading checkpoint shards for {modelname}")
                # If file is split, get the first part's directory to use the same cache location
                weights_path = os.path.join(download_dir, weights_filename)
                if os.path.exists(weights_path):
                    logging.info(f"Found {weights_path}")
                else:
                    # Download and join parts
                    parts = []
                    part_num = 0
                    while True:
                        try:
                            part_path = hf_hub_download(
                                repo_id=repo_id,
                                local_dir=download_dir,
                                revision=revision,
                                filename=f"{weights_filename}.part{part_num}",
                            )
                            parts.append(part_path)
                            part_num += 1
                        except huggingface_hub.errors.EntryNotFoundError:
                            break

                    # Join in the same directory
                    with open(weights_path, 'wb') as outfile:
                        for part in parts:
                            with open(part, 'rb') as infile:
                                while True:
                                    chunk = infile.read(8192 * 1024)
                                    if not chunk:
                                        break
                                    outfile.write(chunk)

                    # Cleaning up the parts
                    for part in parts:
                        try:
                            os.remove(part)
                        except OSError as e:
                            print(f"Error removing {part}: {e}")
                        print("Cleaned up shards, final checkpoint saved to", weights_path)

        return torch.load(weights_path, map_location='cpu', weights_only=False)


HYENA_MODEL_OPTIONS: dict[str, Type[HyenaConfig]] = {
    "1b": Hyena1bConfig,
    "1b_nv": HyenaNV1bConfig,
    "7b": Hyena7bConfig,
    "7b_arc_longcontext": Hyena7bARCLongContextConfig,
    "7b_nv": HyenaNV7bConfig,
    "40b": Hyena40bConfig,
    "40b_arc_longcontext": Hyena40bARCLongContextConfig,
    "40b_nv": HyenaNV40bConfig,
    "test": HyenaTestConfig,
    "test_nv": HyenaNVTestConfig,
}


__all__ = [
    "HyenaConfig",
    "Hyena7bConfig",
    "HyenaNV7bConfig",
    "Hyena1bConfig",
    "HyenaNV1bConfig",
    "Hyena40bConfig",
    "HyenaNV40bConfig",
    "Hyena7bARCLongContextConfig",
    "Hyena40bARCLongContextConfig",
    "HyenaTestConfig",
    "HyenaNVTestConfig",
    "HYENA_MODEL_OPTIONS",
]
