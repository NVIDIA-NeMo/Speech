from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union

import pytorch_lightning as L
import torch
import torch.distributed
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.llm import fn
from nemo.lightning import get_vocab_size, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule

HAVE_TE = True
try:
    import transformer_engine
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

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

    required_keys = set()
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))
    # if self.get_attention_mask_from_fusion:
    #     required_keys.remove('attention_mask')

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def gpt_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


def transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm, qk_layernorm=config.qk_layernorm
    )


def local_layer_spec(config: "GPTConfig") -> ModuleSpec:
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm, qk_layernorm=config.qk_layernorm
    )


def default_layer_spec(config: "GPTConfig") -> ModuleSpec:
    if HAVE_TE:
        return transformer_engine_layer_spec(config)
    else:
        return local_layer_spec(config)


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

    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = default_layer_spec
    forward_step_fn: Callable = gpt_forward_step
    data_step_fn: Callable = gpt_data_step

    def configure_model(self, tokenizer) -> "MCoreGPTModel":
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)

        return MCoreGPTModel(
            self,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )


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
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
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


def get_batch_on_this_context_parallel_rank(batch):
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
