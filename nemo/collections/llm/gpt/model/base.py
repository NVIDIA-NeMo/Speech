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

import contextlib
import inspect
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import lightning.pytorch as L
import torch
import torch.distributed
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention as MCoreDotProductAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_batch_on_this_cp_rank
from torch import nn

from nemo.collections.llm import fn
from nemo.lightning import get_vocab_size, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging
from nemo.utils.import_utils import safe_import

_, HAVE_TE = safe_import("transformer_engine")

# Gradient accumulation fusion may be enabled if available, for more information see:
# https://github.com/NVIDIA/Megatron-LM/blob/01945b98d1ea3a2acb5e8301e181a328104f4856/megatron/core/tensor_parallel/layers.py#L575
# TODO: Clean this up with a getter and install instructions
_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda  # noqa: F401  # pylint: disable=unused-import
except ImportError:
    _grad_accum_fusion_available = False

if TYPE_CHECKING:
    from transformers import GenerationConfig

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def gpt_data_step(dataloader_iter, use_mtp=False) -> dict[str, torch.Tensor]:
    """Process a single batch of data from the dataloader iterator.

    This function handles the data loading step for GPT models, managing
    pipeline parallelism by distributing data appropriately across pipeline stages.

    Args:
        dataloader_iter: Iterator over the dataloader
        use_mtp: Whether the Multi-Token Prediction Module is used. Input needs to be passed
                 into the last ppieline stage if mtp is used.

    Returns:
        dict[str, torch.Tensor]: Processed batch with required tensors moved to appropriate devices
    """
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    if "cu_seqlens" in _batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    if parallel_state.is_pipeline_first_stage() or use_mtp:
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_cp_rank(_batch_required_keys)

    return output


def gpt_forward_step(model, batch) -> torch.Tensor:
    """Execute a forward step for the GPT model.

    This function prepares the arguments needed for the model's forward pass
    and handles both normal and packed sequence processing.

    Args:
        model: The GPT model
        batch: The input batch containing tokens, positions, and other required inputs

    Returns:
        torch.Tensor: Output tensor from the model forward pass
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
    }

    if "attention_mask" not in batch:
        assert (
            HAVE_TE
        ), "The dataloader did not provide an attention mask, however Transformer Engine was not detected. \
            This requires Transformer Engine's implementation of fused or flash attention."
    else:
        forward_args["attention_mask"] = batch["attention_mask"]

    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)

    return model(**forward_args)


def transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """Create a Transformer Engine layer specification based on the provided config.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for Transformer Engine based layers
    """
    from megatron.core.models.gpt import gpt_layer_specs

    kwargs = {
        "num_experts": config.num_moe_experts,
        "moe_grouped_gemm": config.moe_grouped_gemm,
        "qk_layernorm": config.qk_layernorm,
        "fp8": bool(config.num_moe_experts and (config.fp8 is not None)),
    }
    if getattr(config, "use_transformer_engine_op_fuser", None) is not None:
        kwargs["use_te_op_fuser"] = config.use_transformer_engine_op_fuser
    return gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec(**kwargs)


def transformer_engine_full_layer_spec(config: "GPTConfig", vp_stage: Optional[int] = None) -> ModuleSpec:
    """Create a full Transformer Engine layer specification with autocast support.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for full TE layers
    """
    from nemo.collections.nlp.models.language_modeling.megatron.gpt_full_te_layer_autocast_spec import (
        get_gpt_full_te_layer_autocast_spec,
    )

    return get_gpt_full_te_layer_autocast_spec(transformer_config=config, vp_stage=vp_stage)


def local_layer_spec(config: "GPTConfig") -> ModuleSpec:
    """Create a local layer specification without Transformer Engine.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for local implementation layers
    """
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        normalization=config.normalization,
    )


def default_layer_spec(config: "GPTConfig", vp_stage: Optional[int] = None) -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability.

    Uses Transformer Engine specs if available, otherwise falls back to local implementation.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: The selected module specification
    """
    if HAVE_TE:
        if config.use_transformer_engine_full_layer_spec:
            return transformer_engine_full_layer_spec(config, vp_stage=vp_stage)
        else:
            return transformer_engine_layer_spec(config)
    else:
        return local_layer_spec(config)


def mtp_block_spec(config: "GPTConfig", vp_stage: Optional[int] = None) -> Optional[ModuleSpec]:
    """Pass in the MTP block spec if model has MTP layers.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: The MTP module specification
    """
    if getattr(config, "mtp_num_layers", None):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

        if isinstance(config.transformer_layer_spec, Callable):
            if 'vp_stage' in inspect.signature(config.transformer_layer_spec).parameters:
                spec = config.transformer_layer_spec(config, vp_stage=vp_stage)
            else:
                spec = config.transformer_layer_spec(config)
        else:
            spec = config.transformer_layer_spec
        return get_gpt_mtp_block_spec(config, spec, use_transformer_engine=HAVE_TE, vp_stage=vp_stage)
    else:
        return None


def torch_dtype_from_mcore_config(config: TransformerConfig) -> torch.dtype:
    """Extract the appropriate torch dtype from a Megatron Core configuration.

    Args:
        config: Megatron Core Transformer configuration

    Returns:
        torch.dtype: The appropriate torch dtype (float16, bfloat16, or float32)
    """
    if config.fp16:
        return torch.float16
    elif config.bf16:
        return torch.bfloat16
    else:
        return torch.float


def torch_dtype_from_dict_config(config: dict[str, Any]) -> torch.dtype:
    """Extract the appropriate torch dtype from a dictionary configuration.

    Args:
        config: Dictionary containing configuration parameters

    Returns:
        torch.dtype: The appropriate torch dtype (float16, bfloat16, or float32)
    """
    if config["fp16"]:
        return torch.float16
    elif config["bf16"]:
        return torch.bfloat16
    else:
        return torch.float


@dataclass
class GPTConfig(TransformerConfig, io.IOMixin):
    """Configuration class for GPT models.

    Extends TransformerConfig with additional parameters specific to GPT models
    and provides utility methods for model configuration.
    """

    # From megatron.core.models.gpt.gpt_model.GPTModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024
    attention_softmax_in_fp32: bool = False
    masked_softmax_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    gradient_accumulation_fusion: bool = _grad_accum_fusion_available
    deallocate_pipeline_outputs: bool = True
    scatter_embedding_sequence_parallel: bool = True
    tp_only_amax_red: bool = False

    use_transformer_engine_full_layer_spec: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = default_layer_spec

    forward_step_fn: Callable = gpt_forward_step
    data_step_fn: Callable = gpt_data_step
    generation_config: Optional["GenerationConfig"] = None

    vocab_size: Optional[int] = None
    tp_comm_overlap_cfg: Optional[Union[str, dict[str, Any]]] = None

    def configure_model(self, tokenizer, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core GPT model based on this configuration.

        Args:
            tokenizer: Tokenizer used with the model
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        # Enable per-Transformer layer cuda graph.
        if self.enable_cuda_graph and self.cuda_graph_scope != "full_iteration":
            assert HAVE_TE, "Transformer Engine is required for cudagraphs."
            assert getattr(self, "use_te_rng_tracker", False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(
            self, "account_for_loss_in_pipeline_split", False
        )
        is_pipeline_asymmetric |= (
            getattr(self, "num_layers_in_first_pipeline_stage", None)
            or getattr(self, "num_layers_in_last_pipeline_stage", None)
        ) is not None
        is_flexible_pp_layout = is_pipeline_asymmetric or (
            getattr(self, "pipeline_model_parallel_layout", None) is not None
        )
        if vp_size and not is_flexible_pp_layout:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        import inspect

        from megatron.core import parallel_state

        # During fake lightning initialization, pass 0 to bypass the assertion that vp_stage must be
        # non-None when using virtual pipeline model parallelism
        vp_stage = vp_stage or 0

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            # Check if the transformer_layer_spec function accepts vp_stage parameter
            if 'vp_stage' in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        if self.vocab_size is not None:
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logging.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)
        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(torch.device, device='meta')

        if 'mtp_block_spec' in inspect.signature(MCoreGPTModel.__init__).parameters:
            kwargs = {"mtp_block_spec": mtp_block_spec(self, vp_stage=vp_stage)}
        else:
            kwargs = {}

        if self.attention_backend == AttnBackend.local:
            if hasattr(transformer_layer_spec, 'submodules'):
                transformer_layer_spec.submodules.self_attention.submodules.core_attention = MCoreDotProductAttention
        with model_init_device_context():
            model = MCoreGPTModel(
                self,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process
                or parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
                post_process=post_process
                or parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                vp_stage=vp_stage,
                **kwargs,
            )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if HAVE_TE and self.use_transformer_engine_full_layer_spec:
            # Copied from:
            # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_tensor_parallel_group"):
                        tp_group = parallel_state.get_tensor_model_parallel_group()
                        child.set_tensor_parallel_group(tp_group)

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_stream = torch.cuda.Stream()
                for module in self.get_model_module_list():
                    for index, child in enumerate(module.modules()):
                        if index == 0:
                            continue
                        if hasattr(child, "set_context_parallel_group"):
                            child.set_context_parallel_group(
                                parallel_state.get_context_parallel_group(),
                                parallel_state.get_context_parallel_global_ranks(),
                                cp_stream,
                            )

        return model


@dataclass
class GPTConfig126M(GPTConfig):
    """Configuration for a 126M parameter GPT model.

    Predefined configuration for a small GPT model with 12 layers,
    768 hidden size, and 12 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig5B(GPTConfig):
    """Configuration for a 5B parameter GPT model.

    Predefined configuration for a medium-sized GPT model with 24 layers,
    4096 hidden size, and 32 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 24
    hidden_size: int = 4096
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig7B(GPTConfig):
    """Configuration for a 7B parameter GPT model.

    Predefined configuration for a medium-sized GPT model with 32 layers,
    4096 hidden size, and 32 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 10880
    num_attention_heads: int = 32
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig20B(GPTConfig):
    """Configuration for a 20B parameter GPT model.

    Predefined configuration for a large GPT model with 44 layers,
    6144 hidden size, and 48 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 44
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig40B(GPTConfig):
    """Configuration for a 40B parameter GPT model.

    Predefined configuration for a large GPT model with 48 layers,
    8192 hidden size, and 64 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 48
    hidden_size: int = 8192
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 64
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig175B(GPTConfig):
    """Configuration for a 175B parameter GPT model.

    Predefined configuration for a massive GPT model with 96 layers,
    12288 hidden size, and 96 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 96
    hidden_size: int = 12288
    ffn_hidden_size: int = 49152
    num_attention_heads: int = 96
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True
    layernorm_zero_centered_gamma: bool = True


class GPTModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    """GPT model implementation using Megatron Core and PyTorch Lightning.

    This class provides a high-level interface for training and using GPT models
    with proper integration with NeMo's infrastructure.
    """

    def __init__(
        self,
        config: GPTConfig,
        # TODO: Add transformer_layer_spec when we update mcore
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
        model_context_managers: Optional[list] = [],
    ):
        """Initialize the GPT model.

        Args:
            config: Configuration for the GPT model
            optim: Optional optimizer module
            tokenizer: Optional tokenizer specification
            model_transform: Optional function to transform the model after initialization
            model_context_managers: Optional list of context managers to apply when configuring and instantiating
                the model.
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self.model_context_managers = model_context_managers
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        """Configure the underlying model if not already configured.

        This method ensures the model is instantiated from the configuration.
        """
        from nemo.collections.llm.modelopt.model_utils import restore_modelopt_state

        if not hasattr(self, "module"):
            with contextlib.ExitStack() as stack:
                # Apply requested context managers for this block
                for cm in self.model_context_managers:
                    stack.enter_context(cm)

                self.module = self.config.configure_model(self.tokenizer, vp_stage=vp_stage)

            # Restore ModelOpt state if it exists.
            # NOTE: Also called in MegatronStrategy.load_checkpoint but we do it for GPTModel here first,
            # for transformations which add new parameters to the model that need to be included in the optimizer.
            # TODO: Add to other models when needed.
            restore_modelopt_state(self.module, trainer=self._trainer)  # `self.trainer` throws exception if not set

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Forward pass through the GPT model.

        Args:
            input_ids: Input token IDs
            position_ids: Position IDs for the input
            attention_mask: Optional attention mask
            labels: Optional labels for computing loss
            decoder_input: Optional decoder input
            inference_context: Optional parameters for inference
            packed_seq_params: Optional parameters for packed sequence processing

        Returns:
            torch.Tensor: Output tensor from the model
        """
        extra_kwargs = {"packed_seq_params": packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            **extra_kwargs,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> dict[str, torch.Tensor]:
        """Process a batch of data from the dataloader.

        Args:
            dataloader_iter: Iterator over the dataloader

        Returns:
            dict[str, torch.Tensor]: Processed batch
        """
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        """Execute a forward step using the provided batch.

        Args:
            batch: Input batch

        Returns:
            torch.Tensor: Output from the forward pass
        """
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        """Execute a training step.

        Args:
            batch: Input batch
            batch_idx: Optional batch index

        Returns:
            torch.Tensor: Loss value
        """
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        """Execute a validation step.

        Args:
            batch: Input batch
            batch_idx: Optional batch index

        Returns:
            torch.Tensor: Loss value
        """
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    def get_inference_wrapper(
        self,
        params_dtype: torch.dtype,
        inference_batch_times_seqlen_threshold: int,
        inference_max_seq_length: int = 2560,
    ) -> GPTInferenceWrapper:
        """Get an inference wrapper for the model.

        Creates and configures a GPTInferenceWrapper around the model for efficient inference.

        Args:
            params_dtype: Data type for parameters
            inference_batch_times_seqlen_threshold: Threshold for optimizing inference
            inference_max_seq_length: Maximum sequence length for inference (prefill and decode)

        Returns:
            GPTInferenceWrapper: Wrapped model for inference
        """
        # This is to get the MCore model required in GPTInferenceWrapper.
        mcore_model = self.module
        while mcore_model:
            if type(mcore_model) is MCoreGPTModel:
                break
            mcore_model = getattr(mcore_model, "module", None)
        if mcore_model is None or type(mcore_model) is not MCoreGPTModel:
            raise ValueError("Exact McoreGPTModel instance not found in the model structure.")

        vocab_size = None
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
        else:
            raise ValueError(
                "Unable to find vocab size."
                " Either pass in a tokenizer with vocab size, or set vocab size in the model config"
            )

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
            inference_max_seq_length=inference_max_seq_length,
        )

        model_inference_wrapper = GPTInferenceWrapper(mcore_model, inference_wrapper_config)
        return model_inference_wrapper

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        """Get the loss reduction module for training.

        Returns:
            MaskedTokenLossReduction: Loss reduction module for training
        """
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        """Get the loss reduction module for validation.

        Returns:
            MaskedTokenLossReduction: Loss reduction module for validation
        """
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True)

        return self._validation_loss_reduction


def get_packed_seq_params(batch):
    """Extract packed sequence parameters from the batch.

    Creates and returns a PackedSeqParams object with appropriate parameters
    for packed sequence processing.

    Args:
        batch: Input batch containing packed sequence information

    Returns:
        PackedSeqParams: Parameters for packed sequence processing
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = batch["cu_seqlens"].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if (cu_seqlens_argmin := batch.get("cu_seqlens_argmin", None)) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )


__all__ = [
    "GPTModel",
    "GPTConfig",
    "gpt_data_step",
    "gpt_forward_step",
    "transformer_engine_layer_spec",
    "local_layer_spec",
]
