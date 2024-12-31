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
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union

import lightning.pytorch as L
import torch
import torch.distributed
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
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
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def gpt_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
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
    if 'cu_seqlens' in _batch:
        required_device_keys.add('cu_seqlens')
        required_host_keys.add('cu_seqlens_argmin')
        required_host_keys.add('max_seqlen')

    if parallel_state.is_pipeline_first_stage():
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
    output = get_batch_on_this_context_parallel_rank(_batch_required_keys)

    return output


def gpt_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
    }

    if 'attention_mask' not in batch:
        assert (
            HAVE_TE
        ), "The dataloader did not provide an attention mask, however Transformer Engine was not detected. \
            This requires Transformer Engine's implementation of fused or flash attention."
    else:
        forward_args["attention_mask"] = batch['attention_mask']

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


def transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        fp8=bool(config.num_moe_experts and (config.fp8 is not None)),
    )


def transformer_engine_full_layer_spec(config: "GPTConfig") -> ModuleSpec:
    from nemo.collections.nlp.models.language_modeling.megatron.gpt_full_te_layer_autocast_spec import (
        get_gpt_full_te_layer_autocast_spec,
    )

    return get_gpt_full_te_layer_autocast_spec(transformer_config=config)


def local_layer_spec(config: "GPTConfig") -> ModuleSpec:
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm, qk_layernorm=config.qk_layernorm
    )


def default_layer_spec(config: "GPTConfig") -> ModuleSpec:
    if HAVE_TE:
        if config.use_transformer_engine_full_layer_spec:
            return transformer_engine_full_layer_spec(config)
        else:
            return transformer_engine_layer_spec(config)
    else:
        return local_layer_spec(config)


def torch_dtype_from_mcore_config(config: TransformerConfig):
    if config.fp16:
        return torch.float16
    elif config.bf16:
        return torch.bfloat16
    else:
        return torch.float


@dataclass
class GPTConfig(TransformerConfig, io.IOMixin):
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

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "MCoreGPTModel":
        if self.enable_cuda_graph:
            assert HAVE_TE, "Transformer Engine is required for cudagraphs."
            assert getattr(self, 'use_te_rng_tracker', False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)

        if hasattr(self, 'vocab_size'):
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logging.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)

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
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
            post_process=post_process or parallel_state.is_pipeline_last_stage(),
            scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
        )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if HAVE_TE and self.use_transformer_engine_full_layer_spec:
            # Copied from: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
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
    def __init__(
        self,
        config: GPTConfig,
        # TODO: Add transformer_layer_spec when we update mcore
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        extra_kwargs = {'packed_seq_params': packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            **extra_kwargs,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    def get_inference_wrapper(self, params_dtype, inference_batch_times_seqlen_threshold) -> torch.Tensor:
        # This is to get the MCore model required in GPTInferenceWrapper.
        mcore_model = self.module
        while mcore_model:
            if type(mcore_model) is MCoreGPTModel:
                break
            mcore_model = getattr(mcore_model, "module", None)
        if mcore_model is None or type(mcore_model) is not MCoreGPTModel:
            raise ValueError("Exact McoreGPTModel instance not found in the model structure.")

        vocab_size = None
        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.config, 'vocab_size'):
            vocab_size = self.config.vocab_size
        else:
            raise ValueError(
                'Unable to find vocab size. Either pass in a tokenizer with vocab size, or set vocab size in the model config'
            )

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
        )

        model_inference_wrapper = GPTInferenceWrapper(mcore_model, inference_wrapper_config)
        return model_inference_wrapper

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True)

        return self._validation_loss_reduction


def get_batch_on_this_context_parallel_rank(batch) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    if (cp_size := parallel_state.get_context_parallel_world_size()) > 1:
        num_valid_tokens_in_ub = None
        if 'loss_mask' in batch and batch['loss_mask'] is not None:
            num_valid_tokens_in_ub = batch['loss_mask'].sum()

        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2
                _val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                    non_blocking=True
                )
                _val = _val.index_select(seq_dim, index)
                _val = _val.view(*val.shape[0:seq_dim], -1, *_val.shape[(seq_dim + 2) :])
                batch[key] = _val
        batch['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub
    return batch


def get_packed_seq_params(batch):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = batch['cu_seqlens'].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if (cu_seqlens_argmin := batch.get('cu_seqlens_argmin', None)) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch['max_seqlen'].squeeze() if 'max_seqlen' in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )


__all__ = [
    "GPTModel",
    "GPTConfig",
    "gpt_data_step",
    "gpt_forward_step",
    "transformer_engine_layer_spec",
    "local_layer_spec",
]
