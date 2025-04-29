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
from typing import TYPE_CHECKING, Annotated, Callable, Dict, Literal, Optional, Union

import einops
import lightning.pytorch as L
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from torch import Tensor, nn

import nemo.collections.llm.gpt.model.base as GPTBase
from nemo.collections.llm.bert.loss import BERTInBatchExclusiveHardNegativesRankingLoss, HardNegativeRankingLoss
from nemo.collections.llm.gpt.model import GPTConfig
from nemo.collections.llm.gpt.model.llama import (
    HFLlamaImporter,
    Llama32Config1B,
    Llama32Config3B,
    LlamaConfig,
    LlamaModel,
)
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging
from nemo.utils.import_utils import safe_import

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
    from nemo.collections.llm.gpt.model.hf_llama_embedding import LlamaBidirectionalModel

_, HAVE_TE = safe_import("transformer_engine")


def _local_layer_spec(config: "GPTConfig") -> ModuleSpec:
    gpt_layer_spec = GPTBase.local_layer_spec(config)
    gpt_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding
    return gpt_layer_spec


def _transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
    gpt_layer_spec = GPTBase.transformer_engine_layer_spec(config)
    gpt_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding
    return gpt_layer_spec


def get_nv_embedding_layer_spec(config):
    """Customized Layer Spec for NV Embedding Llama Model.
    Bidirectional attention is enabled instead of causal masking.
    """
    if HAVE_TE:
        return _transformer_engine_layer_spec(config)
    else:
        return _local_layer_spec(config)


def nv_embedding_data_step(dataloder_iter) -> Dict[str, torch.Tensor]:
    """Setup NVEmbedding Llama Model dataloader batch."""
    batch = next(dataloder_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")

    if parallel_state.is_pipeline_first_stage():
        required_keys.add("input_ids")
        required_keys.add("position_ids")

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = GPTBase.get_batch_on_this_context_parallel_rank(_batch)

    return output


def nv_embedding_forward_step(model: L.LightningModule, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    This subsets the batch keys to the ones actually used by forward pass of the model,
    and then calls the model's forward pass. if "cu_seqsens" are defined in the batch,
    then the packed sequence parameters are also passed to the model for forward pass efficiency.
    """
    forward_args = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "position_ids": batch["position_ids"],
    }
    emb = model.encode(**forward_args)
    return emb


@dataclass
class Llama32EmbeddingConfig1B(Llama32Config1B):
    """Llama3.2 Embedding 1B Config"""

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = get_nv_embedding_layer_spec
    forward_step_fn: Callable = nv_embedding_forward_step
    data_step_fn: Callable = nv_embedding_data_step

    # Training Configs
    truncation_method: Literal["left", "right"] = 'right'
    num_hard_negatives: int = 4
    ce_loss_scale: float = 50
    label_smoothing: float = 0.0
    in_batch_negatives: bool = False
    negative_sample_strategy: Literal["random", "first"] = 'first'
    add_bos: bool = True
    add_eos: bool = False

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "MCoreGPTModel":
        """Configure the NV Embedding Llama3.2 1B Model"""
        model = super().configure_model(tokenizer, pre_process, post_process)
        # post_process need to be overwritten to False after model init because
        # final_layernorm is still needed and it will only be initialized when post_process is True in Mcore.
        # And for forward(), we do not want to run through output_layer thus setting post_process to False.
        model.post_process = False
        return model


@dataclass
class Llama32EmbeddingConfig3B(Llama32Config3B):
    """Llama3.2 Embedding 3B Config"""

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = get_nv_embedding_layer_spec
    forward_step_fn: Callable = nv_embedding_forward_step
    data_step_fn: Callable = nv_embedding_data_step

    # Training Configs
    truncation_method: Literal["left", "right"] = 'right'
    num_hard_negatives: int = 4
    ce_loss_scale: float = 50
    label_smoothing: float = 0.0
    in_batch_negatives: bool = False
    negative_sample_strategy: Literal["random", "first"] = 'first'
    add_bos: bool = True
    add_eos: bool = False

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "MCoreGPTModel":
        """Configure the NV Embedding Llama3.2 3B Model"""
        model = super().configure_model(tokenizer, pre_process, post_process)
        # post_process need to be overwritten to False after model init because
        # final_layernorm is still needed and it will only be initialized when post_process is True in Mcore.
        # And for forward(), we do not want to run through output_layer thus setting post_process to False.
        model.post_process = False
        return model


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor):
    """Average the hidden states on the non-masking tokens."""
    # [sq, b, h] -> [b, sq, h]
    last_hidden_states = einops.rearrange(last_hidden_states, 's b h -> b s h')
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class LlamaEmbeddingModel(LlamaModel):
    """NV Embedding Llama Model"""

    def __init__(
        self,
        config: Annotated[Optional[LlamaConfig], Config[LlamaConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or LlamaConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)

    @property
    def dataset_kwargs(self):
        """Getter for dataset_kwargs from model config"""
        return {
            'num_hard_negatives': self.config.num_hard_negatives,
            'negative_sample_strategy': self.config.negative_sample_strategy,
            'add_bos': self.config.add_bos,
            'add_eos': self.config.add_eos,
        }

    def encode(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        decoder_input: Optional[torch.Tensor] = None,
    ):
        """Generate the embedding for the inputs.
        It runs the forward and apply average pooling on the last hidden states of the model.
        """
        if attention_mask.ndim == 2:
            # extend attention mask to [b, 1, 1, sq]
            # Also convert attention mask to binary
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(1) < 0.5
        elif attention_mask.ndim == 4:
            assert attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1, "Attention mask shape incorrect"
            extended_mask = attention_mask
            # Squeeze attention mask to [b, sq] for averaging pooling later

            attention_mask = extended_mask.squeeze() < 0.5
        else:
            raise ValueError("Attention_mask shape incorrect")

        output = self.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=extended_mask,
            decoder_input=decoder_input,
        )
        embeddings = _average_pool(output, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @property
    def training_loss_reduction(self) -> BERTInBatchExclusiveHardNegativesRankingLoss:  # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            if self.config.in_batch_negatives:
                loss_func = BERTInBatchExclusiveHardNegativesRankingLoss
            else:
                loss_func = HardNegativeRankingLoss
            self._training_loss_reduction = loss_func(
                validation_step=False,
                num_hard_negatives=self.config.num_hard_negatives,
                scale=self.config.ce_loss_scale,
                label_smoothing=self.config.label_smoothing,
            )

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> BERTInBatchExclusiveHardNegativesRankingLoss:  # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            if self.config.in_batch_negatives:
                loss_func = BERTInBatchExclusiveHardNegativesRankingLoss
            else:
                loss_func = HardNegativeRankingLoss
            self._validation_loss_reduction = loss_func(
                validation_step=True,
                num_hard_negatives=self.config.num_hard_negatives,
                scale=self.config.ce_loss_scale,
                label_smoothing=self.config.label_smoothing,
            )

        return self._validation_loss_reduction


@io.model_importer(LlamaEmbeddingModel, "hf")
class LlamaEmbeddingImporter(HFLlamaImporter):
    """HF Importer for Llama Embedding Model"""

    def init(self) -> LlamaEmbeddingModel:
        return LlamaEmbeddingModel(self.config, tokenizer=self.tokenizer)

    @property
    def config(self) -> Llama32Config1B:
        # pylint : disable=C0116
        from transformers import LlamaConfig as HFLlamaConfig

        source = HFLlamaConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = Llama32EmbeddingConfig1B(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(LlamaEmbeddingModel, "hf")
class LlamaEmbeddingExporter(io.ModelConnector[LlamaEmbeddingModel, "LlamaBidirectionalModel"]):
    """HF Exporter for NV Embedding Llama Model.
    Note that NV Embedding LLama uses customized LlamaBidirectionalConfig config.
    """

    def init(self, dtype=torch.bfloat16) -> "LlamaBidirectionalModel":
        from transformers.modeling_utils import no_init_weights

        from nemo.collections.llm.gpt.model.hf_llama_embedding import LlamaBidirectionalModel

        LlamaBidirectionalModel.register_for_auto_class("AutoModel")
        with no_init_weights():
            return LlamaBidirectionalModel._from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        source, _ = self.nemo_load(str(self))
        source_dtype = source.module.embedding.word_embeddings.weight.dtype
        target = self.init(source_dtype)
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        try:
            tokenizer = self.tokenizer.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = source.config.truncation_method

            tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    @property
    def config(self):
        """Get HF NV Embedding Llama Config."""
        source: LlamaConfig = io.load_context(str(self), subpath="model.config")

        from nemo.collections.llm.gpt.model.hf_llama_embedding import LlamaBidirectionalConfig

        LlamaBidirectionalConfig.register_for_auto_class("AutoConfig")
        return LlamaBidirectionalConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
        )

    def convert_state(self, source, target):
        """Convert NeMo State dict to HF."""
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "norm.weight",
        }
        transforms = [
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "layers.*.self_attn.q_proj.weight",
                    "layers.*.self_attn.k_proj.weight",
                    "layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("layers.*.mlp.gate_proj.weight", "layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="embed_tokens.weight",
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
        """Get NeMo Tokenizer"""
        return io.load_context(str(self), subpath="model").tokenizer


__all__ = [
    "Llama32EmbeddingConfig1B",
    "LlamaEmbeddingModel",
]
