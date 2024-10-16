import copy
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
    from megatron.core.models.T5.t5_model import T5Model as MCoreT5Model

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def t5_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import AttnMaskType
    from nemo.collections.nlp.modules.common.megatron.utils import build_attention_mask_3d

    batch = next(dataloader_iter)

    _batch: dict
    # TODO: to fix for running inferencing
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    # if Dataset object is NeMo 1.0's T5SFTDataset (e.g. when finetuning with SQUAD)
    if 'enc_dec_mask' not in _batch:
        encoder_attn_mask_3d = build_attention_mask_3d(_batch['enc_mask'], _batch['enc_mask'], AttnMaskType.padding)
        decoder_attn_mask_3d = build_attention_mask_3d(_batch['dec_mask'], _batch['dec_mask'], AttnMaskType.causal)
        enc_dec_attn_mask_3d = build_attention_mask_3d(_batch['dec_mask'], _batch['enc_mask'], AttnMaskType.padding)
        _batch['enc_mask'] = encoder_attn_mask_3d
        _batch['dec_mask'] = decoder_attn_mask_3d
        _batch['enc_dec_mask'] = enc_dec_attn_mask_3d

    # if Dataset object is Mcore T5 dataset (e.g. pretraining)
    else:
        # convert attention mask values from int to True/False
        _batch['enc_mask'] = _batch['enc_mask'] < 0.5
        _batch['dec_mask'] = _batch['dec_mask'] < 0.5
        _batch['enc_dec_mask'] = _batch['enc_dec_mask'] < 0.5

    required_keys = set()
    required_keys.update(["enc_mask", "dec_mask", "enc_dec_mask"])
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("text_enc", "text_dec"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    output = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}

    return output


def t5_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "encoder_input_ids": batch["text_enc"],
        "decoder_input_ids": batch["text_dec"],
        "encoder_attn_mask": batch["enc_mask"],
        "decoder_attn_mask": batch["dec_mask"],
        "encoder_decoder_attn_mask": batch["enc_dec_mask"],
        "lm_labels": batch["labels"],
    }

    return model(**forward_args)


def transformer_engine_layer_spec(encoder_config: "T5Config", decoder_config: "T5Config") -> ModuleSpec:
    from megatron.core.models.T5.t5_spec import (
        get_t5_decoder_with_transformer_engine_block_spec,
        get_t5_encoder_with_transformer_engine_block_spec,
    )

    en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(encoder_config.num_layers)
    de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(decoder_config.num_layers)

    return [en_block_spec, de_block_spec]


def local_layer_spec(encoder_config: "T5Config", decoder_config: "T5Config") -> ModuleSpec:
    from megatron.core.models.T5.t5_spec import (
        get_t5_decoder_with_local_block_spec,
        get_t5_encoder_with_local_block_spec,
    )

    en_block_spec = get_t5_encoder_with_local_block_spec(encoder_config.num_layers)
    de_block_spec = get_t5_decoder_with_local_block_spec(decoder_config.num_layers)

    return [en_block_spec, de_block_spec]


def default_layer_spec(encoder_config: "T5Config", decoder_config: "T5Config") -> ModuleSpec:
    if HAVE_TE:
        return transformer_engine_layer_spec(encoder_config, decoder_config)
    else:
        return local_layer_spec(encoder_config, decoder_config)


@dataclass
class T5Config(TransformerConfig, io.IOMixin):
    # From megatron.core.models.t5.t5_model.T5Model
    encoder_num_layers: int = None
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    max_position_embeddings: int = 512
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    encoder_pipeline_model_parallel_size: int = 0
    attention_softmax_in_fp32: float = False
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    deallocate_pipeline_outputs: bool = True
    pipeline_model_parallel_split_rank: int = 0
    num_moe_experts: int = 1
    recompute_num_layers: int = 1
    distribute_saved_activations: bool = False
    enable_autocast: bool = False

    transformer_layer_spec: Union[ModuleSpec, Callable[["T5Config"], ModuleSpec]] = default_layer_spec
    forward_step_fn: Callable = t5_forward_step
    data_step_fn: Callable = t5_data_step

    def configure_model(self, tokenizer) -> "MCoreT5Model":
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state
        from megatron.core.models.T5.t5_model import T5Model as MCoreT5Model

        encoder_config = copy.deepcopy(self)
        encoder_config.num_layers = self.encoder_num_layers
        if self.pipeline_model_parallel_size > 1:
            assert self.encoder_pipeline_model_parallel_size > 0, "Need to know how to shard the encoder & decoder."
            encoder_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(encoder_config=encoder_config, decoder_config=self)

        model = MCoreT5Model(
            config=self,
            encoder_config=encoder_config,
            transformer_encoder_layer_spec=transformer_layer_spec[0],
            transformer_decoder_layer_spec=transformer_layer_spec[1],
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.max_position_embeddings,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )

        return model


class T5Model(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: T5Config,
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
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
        decoder_attn_mask: torch.Tensor,
        encoder_decoder_attn_mask: torch.Tensor,
        lm_labels: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:

        output_tensor = self.module(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_decoder_attn_mask=encoder_decoder_attn_mask,
            lm_labels=lm_labels,
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


__all__ = [
    "T5Model",
    "T5Config",
    "t5_data_step",
    "t5_forward_step",
    "transformer_engine_layer_spec",
    "local_layer_spec",
]
