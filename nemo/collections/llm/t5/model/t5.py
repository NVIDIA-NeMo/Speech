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

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union

import lightning.pytorch as L
import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig

from megatron.core.models.T5.t5_model import T5Model as MCoreT5Model
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from transformers import T5Config as HFT5Config
from transformers import T5ForConditionalGeneration

from nemo.collections.llm import fn
from nemo.lightning import get_vocab_size, io, teardown
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule

HAVE_TE = True
try:
    import transformer_engine
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def t5_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    """Processing data for one step of T5 model"""

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

    # work for both mcore's T5 pre-train dataset object, and NeMo's T5SFTDataset dataset
    enc_mask = _batch['enc_mask'] < 0.5
    dec_mask = _batch['dec_mask'] < 0.5
    # process for Flash/Fused
    enc_mask = enc_mask.unsqueeze(1).unsqueeze(1)
    dec_mask = dec_mask.unsqueeze(1).unsqueeze(1)
    enc_dec_mask = (
        dec_mask,
        enc_mask,
    )
    # set dec_mask to None because decoder uses AttnMaskType.causal
    dec_mask = None
    _batch['enc_mask'] = enc_mask
    _batch['dec_mask'] = dec_mask
    _batch['enc_dec_mask'] = enc_dec_mask

    # bring to device
    for key in _batch.keys():
        if key == "enc_dec_mask":  # because enc_dec_mask is a tuple
            _batch[key] = (_batch[key][0].cuda(non_blocking=True), _batch[key][1].cuda(non_blocking=True))
        elif key == "dec_mask":  # because dec_mask is a None since decoder uses AttnMaskType.causal
            continue
        else:
            _batch[key] = _batch[key].cuda(non_blocking=True)

    # set up forward arguments for pipeline parallelism
    required_keys = set()
    required_keys.update(["enc_mask", "dec_mask", "enc_dec_mask"])
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("text_enc", "text_dec"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    output = {key: val if key in required_keys else None for key, val in _batch.items()}

    return output


def t5_forward_step(model, batch) -> torch.Tensor:
    """Processing a forward step for T5 model"""
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
    """Spec for T5 when using transformer_engine mcore implementation"""
    from megatron.core.models.T5.t5_spec import (
        get_t5_decoder_with_transformer_engine_block_spec,
        get_t5_encoder_with_transformer_engine_block_spec,
    )

    en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(encoder_config.num_layers)
    de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(decoder_config.num_layers)

    return [en_block_spec, de_block_spec]


def local_layer_spec(encoder_config: "T5Config", decoder_config: "T5Config") -> ModuleSpec:
    """Spec for T5 when using local mcore implementation"""
    from megatron.core.models.T5.t5_spec import (
        get_t5_decoder_with_local_block_spec,
        get_t5_encoder_with_local_block_spec,
    )

    en_block_spec = get_t5_encoder_with_local_block_spec(encoder_config.num_layers)
    de_block_spec = get_t5_decoder_with_local_block_spec(decoder_config.num_layers)

    return [en_block_spec, de_block_spec]


def default_layer_spec(encoder_config: "T5Config", decoder_config: "T5Config") -> ModuleSpec:
    """Set layer spec conditioning on whether transformer_engine is available"""
    if HAVE_TE:
        return transformer_engine_layer_spec(encoder_config, decoder_config)
    else:
        return local_layer_spec(encoder_config, decoder_config)


@dataclass
class T5Config(TransformerConfig, io.IOMixin):
    """Model config for T5 model. Adpated from megatron.core.models.t5.t5_model.T5Model"""

    encoder_num_layers: int = None
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    apply_rope_fusion: bool = True
    max_position_embeddings: int = 512
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 512
    seq_length_dec: int = 128
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
        """Setup the T5 Model based on config definition."""

        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state

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


@dataclass
class T5Config220M(T5Config):
    """
    NeMo's T5 model variant
    https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/main/launcher_scripts/conf/training/t5/220m.yaml
    """

    num_layers: int = 12
    encoder_num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12


@dataclass
class T5Config3B(T5Config):
    """Config for 3B T5 model"""

    num_layers: int = 24
    encoder_num_layers: int = 24
    hidden_size: int = 2048
    ffn_hidden_size: int = 5120
    num_attention_heads: int = 32


@dataclass
class T5Config11B(T5Config):
    """Config for 11B T5 model"""

    num_layers: int = 24
    encoder_num_layers: int = 24
    hidden_size: int = 4096
    ffn_hidden_size: int = 10240
    num_attention_heads: int = 64


class T5Model(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    """T5 Lightning Module"""

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
        """Setup the T5 Model based on config definition."""
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
        """Call the forward method of the underlying model, and return whatever it outputs."""

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

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:  # pylint: disable=C0115,C0116
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:  # pylint: disable=C0115,C0116
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:  # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:  # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    def get_inference_wrapper(self, params_dtype, inference_batch_times_seqlen_threshold) -> torch.Tensor:
        """This is to get the MCore model required in T5InferenceWrapper"""
        mcore_model = self.module
        while mcore_model:
            if type(mcore_model) is MCoreT5Model:
                break
            mcore_model = getattr(mcore_model, "module", None)
        if mcore_model is None or type(mcore_model) is not MCoreT5Model:
            raise ValueError("Exact MCoreT5Model instance not found in the model structure.")

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=self.tokenizer.vocab_size,
        )
        from megatron.core.inference.model_inference_wrappers.t5.t5_inference_wrapper import T5InferenceWrapper

        model_inference_wrapper = T5InferenceWrapper(mcore_model, inference_wrapper_config)
        return model_inference_wrapper

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:  # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:  # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True)

        return self._validation_loss_reduction


@io.model_importer(T5Model, "hf")
class HFT5Importer(io.ModelConnector["T5ForConditionalGeneration", T5Model]):
    """Importer Connector for converting HF Google T5 Model to NeMo"""

    def init(self) -> T5Model:
        return T5Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import T5ForConditionalGeneration

        source = T5ForConditionalGeneration.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)

        self.nemo_save(output_path, trainer)

        print(f"Converted T5 model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Converting HF state dict to NeMo state dict."""
        mapping = {
            "shared.weight": "embedding.word_embeddings.weight",
            "lm_head.weight": "lm_head.output_layer.weight",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": "encoder_relative_pos_emb.relative_attention_bias.weight",
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": "decoder_relative_pos_emb.relative_attention_bias.weight",
            "encoder.block.*.layer.0.layer_norm.weight": "encoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "encoder.block.*.layer.0.SelfAttention.o.weight": "encoder.layers.*.self_attention.linear_proj.weight",
            "encoder.block.*.layer.1.layer_norm.weight": "encoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "encoder.block.*.layer.1.DenseReluDense.wo.weight": "encoder.layers.*.mlp.linear_fc2.weight",
            "encoder.final_layer_norm.weight": "encoder.final_layernorm.weight",
            "decoder.block.*.layer.0.layer_norm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "decoder.block.*.layer.0.SelfAttention.o.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "decoder.block.*.layer.1.layer_norm.weight": "decoder.layers.*.pre_cross_attn_layernorm.weight",
            "decoder.block.*.layer.1.EncDecAttention.q.weight": "decoder.layers.*.cross_attention.linear_q.weight",
            "decoder.block.*.layer.1.EncDecAttention.o.weight": "decoder.layers.*.cross_attention.linear_proj.weight",
            "decoder.block.*.layer.2.layer_norm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "decoder.block.*.layer.2.DenseReluDense.wo.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "decoder.final_layer_norm.weight": "decoder.final_layernorm.weight",
        }
        if getattr(source.config, "tie_word_embeddings", False):
            del mapping["lm_head.weight"]

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _import_encoder_qkv,
                _import_encoder_linear_fc1,
                _import_decoder_qkv,
                _import_decoder_kv,
                _import_decoder_linear_fc1,
            ],
            state_dict_ignored_entries=['output_layer.weight'],
        )

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """Retrieve Tokenizer from HF"""
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        # Set special tokens to match HF
        bos_token = "<pad>"

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)), bos_token=bos_token)

    @property
    def config(self) -> T5Config:
        """Generate NeMo Config based on HF config"""
        from transformers import T5Config as HFT5Config

        source = HFT5Config.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = T5Config(
            num_layers=source.num_layers,
            encoder_num_layers=source.num_decoder_layers,
            hidden_size=source.d_model,
            ffn_hidden_size=source.d_ff,
            kv_channels=source.d_kv,
            num_attention_heads=source.num_heads,
            position_embedding_type="relative",
            relative_attention_num_buckets=source.relative_attention_num_buckets,
            relative_attention_max_distance=source.relative_attention_max_distance,
            activation_func=F.gelu,
            add_bias_linear=False,
            init_method_std=source.initializer_factor,
            normalization="RMSNorm",
            layernorm_epsilon=source.layer_norm_epsilon,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=False,
            bf16=False,
            params_dtype=torch.float32,
            softmax_scale=1.0,
        )

        return output


@io.state_transform(
    source_key=(
        "encoder.block.*.layer.0.SelfAttention.q.weight",
        "encoder.block.*.layer.0.SelfAttention.k.weight",
        "encoder.block.*.layer.0.SelfAttention.v.weight",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_encoder_qkv(ctx: io.TransformCTX, q, k, v):
    # T5 Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_q_tensor_shape)
    v = v.view(*new_q_tensor_shape)

    qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
    for i in range(head_num):
        qkv_weights = torch.cat((qkv_weights, q[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    qkv_weights = qkv_weights.reshape([head_size * (3 * head_num), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "decoder.block.*.layer.0.SelfAttention.q.weight",
        "decoder.block.*.layer.0.SelfAttention.k.weight",
        "decoder.block.*.layer.0.SelfAttention.v.weight",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_decoder_qkv(ctx: io.TransformCTX, q, k, v):
    # T5 Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_q_tensor_shape)
    v = v.view(*new_q_tensor_shape)

    qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
    for i in range(head_num):
        qkv_weights = torch.cat((qkv_weights, q[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    qkv_weights = qkv_weights.reshape([head_size * (3 * head_num), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=(
        "decoder.block.*.layer.1.EncDecAttention.k.weight",
        "decoder.block.*.layer.1.EncDecAttention.v.weight",
    ),
    target_key="decoder.layers.*.cross_attention.linear_kv.weight",
)
def _import_decoder_kv(ctx: io.TransformCTX, k, v):
    # T5 Model does not support GQA
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = k.size()
    new_k_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]

    k = k.view(*new_k_tensor_shape)
    v = v.view(*new_k_tensor_shape)

    kv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
    for i in range(head_num):
        kv_weights = torch.cat((kv_weights, k[i : i + 1, :, :]))
        kv_weights = torch.cat((kv_weights, v[i : i + 1, :, :]))
    kv_weights = kv_weights.reshape([head_size * (2 * head_num), hidden_size])

    return kv_weights


@io.state_transform(
    source_key=(
        "encoder.block.*.layer.1.DenseReluDense.wi_0.weight",
        "encoder.block.*.layer.1.DenseReluDense.wi_1.weight",
    ),
    target_key="encoder.layers.*.mlp.linear_fc1.weight",
)
def _import_encoder_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)


@io.state_transform(
    source_key=(
        "decoder.block.*.layer.2.DenseReluDense.wi_0.weight",
        "decoder.block.*.layer.2.DenseReluDense.wi_1.weight",
    ),
    target_key="decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_decoder_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)


@io.model_exporter(T5Model, "hf")
class HFT5Exporter(io.ModelConnector[T5Model, "T5ForConditionalGeneration"]):
    """Exporter Connector for converting NeMo T5 Model to HF"""

    def init(self) -> "T5ForConditionalGeneration":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return T5ForConditionalGeneration(config=self.config)

    def apply(self, output_path: Path) -> Path:
        source, _ = self.nemo_load(str(self))
        target = self.init()
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        """Convert NeMo state dict to HF style"""
        mapping = {
            "embedding.word_embeddings.weight": "shared.weight",
            "lm_head.output_layer.weight": "lm_head.weight",
            "encoder_relative_pos_emb.relative_attention_bias.weight": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "decoder_relative_pos_emb.relative_attention_bias.weight": "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "encoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "encoder.block.*.layer.0.layer_norm.weight",
            "encoder.layers.*.self_attention.linear_proj.weight": "encoder.block.*.layer.0.SelfAttention.o.weight",
            "encoder.layers.*.mlp.linear_fc1.layer_norm_weight": "encoder.block.*.layer.1.layer_norm.weight",
            "encoder.layers.*.mlp.linear_fc2.weight": "encoder.block.*.layer.1.DenseReluDense.wo.weight",
            "encoder.final_layernorm.weight": "encoder.final_layer_norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "decoder.block.*.layer.0.layer_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "decoder.block.*.layer.0.SelfAttention.o.weight",
            "decoder.layers.*.pre_cross_attn_layernorm.weight": "decoder.block.*.layer.1.layer_norm.weight",
            "decoder.layers.*.cross_attention.linear_q.weight": "decoder.block.*.layer.1.EncDecAttention.q.weight",
            "decoder.layers.*.cross_attention.linear_proj.weight": "decoder.block.*.layer.1.EncDecAttention.o.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "decoder.block.*.layer.2.layer_norm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "decoder.block.*.layer.2.DenseReluDense.wo.weight",
            "decoder.final_layernorm.weight": "decoder.final_layer_norm.weight",
        }

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                _export_encoder_qkv,
                _export_encoder_linear_fc1,
                _export_decoder_qkv,
                _export_decoder_kv,
                _export_decoder_linear_fc1,
            ],
            state_dict_ignored_entries=['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'],
        )

    @property
    def tokenizer(self):
        """Retrieve Tokenizer from HF"""
        # return io.load_context(str(self)).model.tokenizer.tokenizer
        nemo_tokenizer = io.load_context(str(self)).model.tokenizer
        self.bos_id = nemo_tokenizer.bos_id
        self.pad_id = nemo_tokenizer.pad_id

        return nemo_tokenizer.tokenizer

    @property
    def config(self) -> "HFT5Config":
        """Generate NeMo Config based on HF config"""
        source: T5Config = io.load_context(str(self)).model.config

        from transformers import T5Config as HFT5Config

        nemo_tokenizer = io.load_context(str(self)).model.tokenizer
        bos_id = nemo_tokenizer.bos_id
        pad_id = nemo_tokenizer.pad_id
        eos_id = nemo_tokenizer.eos_id

        def round_up_to_divisible(number, divisor):
            import math

            if divisor == 0:
                raise ValueError("Divisor cannot be zero.")
            return int(math.ceil(number / divisor) * divisor)

        return HFT5Config(
            num_layers=source.num_layers,
            num_decoder_layers=source.encoder_num_layers,
            d_model=source.hidden_size,
            d_ff=source.ffn_hidden_size,
            d_kv=source.kv_channels,
            num_heads=source.num_attention_heads,
            relative_attention_num_buckets=source.relative_attention_num_buckets,
            relative_attention_max_distance=source.relative_attention_max_distance,
            initializer_factor=source.init_method_std,
            layer_norm_epsilon=source.layernorm_epsilon,
            vocab_size=round_up_to_divisible(
                self.tokenizer.vocab_size + len(self.tokenizer.additional_special_tokens), 128
            ),
            feed_forward_proj="gated-gelu",
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            decoder_start_token_id=bos_id,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )


@io.state_transform(
    source_key="encoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "encoder.block.*.layer.0.SelfAttention.q.weight",
        "encoder.block.*.layer.0.SelfAttention.k.weight",
        "encoder.block.*.layer.0.SelfAttention.v.weight",
    ),
)
def _export_encoder_qkv(ctx: io.TransformCTX, linear_qkv):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, hidden_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "decoder.block.*.layer.0.SelfAttention.q.weight",
        "decoder.block.*.layer.0.SelfAttention.k.weight",
        "decoder.block.*.layer.0.SelfAttention.v.weight",
    ),
)
def _export_decoder_qkv(ctx: io.TransformCTX, linear_qkv):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, hidden_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


@io.state_transform(
    source_key="decoder.layers.*.cross_attention.linear_kv.weight",
    target_key=(
        "decoder.block.*.layer.1.EncDecAttention.k.weight",
        "decoder.block.*.layer.1.EncDecAttention.v.weight",
    ),
)
def _export_decoder_kv(ctx: io.TransformCTX, linear_kv):
    megatron_config = ctx.source.config

    num_query_groups = megatron_config.num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    kv_total_dim = 2 * num_query_groups

    linear_kv = linear_kv.reshape([kv_total_dim, head_size, hidden_size])
    k_slice = torch.arange(0, kv_total_dim, 2)
    v_slice = torch.arange(1, kv_total_dim, 2)

    k_proj = linear_kv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_kv[v_slice].reshape(-1, hidden_size).cpu()

    return k_proj, v_proj


@io.state_transform(
    source_key="encoder.layers.*.mlp.linear_fc1.weight",
    target_key=(
        "encoder.block.*.layer.1.DenseReluDense.wi_0.weight",
        "encoder.block.*.layer.1.DenseReluDense.wi_1.weight",
    ),
)
def _export_encoder_linear_fc1(linear_fc1):
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)

    return gate_proj, up_proj


@io.state_transform(
    source_key="decoder.layers.*.mlp.linear_fc1.weight",
    target_key=(
        "decoder.block.*.layer.2.DenseReluDense.wi_0.weight",
        "decoder.block.*.layer.2.DenseReluDense.wi_1.weight",
    ),
)
def _export_decoder_linear_fc1(linear_fc1):
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)

    return gate_proj, up_proj


__all__ = [
    "T5Model",
    "T5Config",
    "t5_data_step",
    "t5_forward_step",
    "transformer_engine_layer_spec",
    "local_layer_spec",
]
