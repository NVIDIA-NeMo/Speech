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
from typing import TYPE_CHECKING, Annotated, Callable, Dict, Literal, Optional, Tuple, Union

import einops
import pytorch_lightning as L
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.module import Float16Module, float16_to_fp32, fp32_to_float16
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_batch_on_this_cp_rank
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.gpt.model.llama import HFLlamaImporter, Llama32Config1B
from nemo.collections.llm.gpt.model.llama_embedding import LlamaEmbeddingExporter
from nemo.collections.llm.gpt.model.llama_embedding import (
    get_nv_embedding_layer_spec as bidirectional_attention_layer_spec,
)
from nemo.collections.llm.utils import Config, is_safe_repo
from nemo.lightning import OptimizerModule, io
from nemo.lightning.io.state import TransformFns
from nemo.lightning.megatron_parallel import DDP, MegatronLossReduction
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from transformers import AutoModelForSequenceClassification

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
    from nemo.collections.llm.gpt.model.llama import LlamaConfig


def reranker_data_step(dataloder_iter) -> Dict[str, torch.Tensor]:
    """Setup Reranker dataloader batch."""
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
    output = get_batch_on_this_cp_rank(_batch)

    return output


def reranker_forward_step(model: L.LightningModule, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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
    score = model.forward(**forward_args)
    return score


@dataclass
class ReRankerBaseConfig:
    """
    Base config for Reranker Models Training configs
    """

    # Training Configs
    truncation_method: Literal["left", "right"] = 'right'
    num_hard_negatives: int = 4
    ce_loss_scale: float = 50
    label_smoothing: float = 0.0
    in_batch_negatives: bool = False
    negative_sample_strategy: Literal["random", "first"] = 'first'
    add_bos: bool = True
    add_eos: bool = False
    pool_type: Optional[Literal["cls", "avg", "last", "weighted_avg"]] = "avg"
    temperature: float = 1.0


@dataclass
class Llama32Reranker1BConfig(Llama32Config1B, ReRankerBaseConfig):
    """Config for Llama32Reranker1B model"""

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = bidirectional_attention_layer_spec
    forward_step_fn: Callable = reranker_forward_step
    data_step_fn: Callable = reranker_data_step
    importer_cls: io.ModelConnector = HFLlamaImporter
    exporter_cls: io.ModelConnector = LlamaEmbeddingExporter

    def configure_model(self, tokenizer, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure the Reranker Model"""
        model = super().configure_model(tokenizer, pre_process, post_process, vp_stage)
        # post_process need to be overwritten to False after model init because
        # final_layernorm is still needed and it will only be initialized when post_process is True in Mcore.
        # And for forward(), we do not want to run through output_layer thus setting post_process to False.
        model.post_process = False
        return model


class Llama32Reranker500MConfig(Llama32Reranker1BConfig):
    """Config for Llama32Reranker500M model"""

    num_layers: int = 8


class ReRankerModel(GPTModel):
    """Base model for Reranking that extends GPTModel with reranking-specific functionality."""

    def __init__(
        self,
        config: Annotated[Optional[GPTConfig], Config[GPTConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or Llama32Reranker1BConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )

    @property
    def dataset_kwargs(self):
        """Getter for dataset_kwargs from model config."""
        return {
            'num_hard_negatives': self.config.num_hard_negatives,
            'negative_sample_strategy': self.config.negative_sample_strategy,
            'add_bos': self.config.add_bos,
            'add_eos': self.config.add_eos,
        }

    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        """Configure the underlying model if not already configured.

        This method ensures the model is instantiated from the configuration.
        """
        assert (
            self.config.pool_type in ["cls", "avg", "last", "weighted_avg"] or self.config.pool_type is None
        ), f"Invalid pool type: {self.config.pool_type} should be in [cls, avg, last, weighted_avg] or None"

        super().configure_model(vp_stage)
        # TODO: handle PP, all args
        self.module.score = ColumnParallelLinear(
            self.config.hidden_size,
            1,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=True,
            gather_output=True,
        )

    def pool(self, last_hidden_states, attention_mask):
        """Pool the hidden states based on the configured pooling strategy.

        Args:
            last_hidden_states: The hidden states from the transformer
            attention_mask: The attention mask for the input

        Returns:
            The pooled embeddings
        """
        # [sq, b, h] -> [b, sq, h]
        last_hidden_states = einops.rearrange(last_hidden_states, 's b h -> b s h')
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        pool_type = self.config.pool_type
        if pool_type == "avg":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif pool_type == "weighted_avg":
            emb = last_hidden.sum(dim=1)
        elif pool_type == "cls":
            emb = last_hidden[:, 0]
        elif pool_type == "last":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                emb = last_hidden[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden.shape[0]
                emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")
        return emb

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        decoder_input: Optional[torch.Tensor] = None,
    ):
        """Forward pass of the reranker model.

        Args:
            input_ids: Input token IDs
            position_ids: Position IDs for the input
            attention_mask: Attention mask for the input
            decoder_input: Optional decoder input

        Returns:
            The pooled logits
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

        output = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=extended_mask,
            decoder_input=decoder_input,
        )

        pooled_hidden_states = self.pool(output, attention_mask)

        # output and pooled_hidden_states are FP32 during training
        # (because the Float16Module wrapper converts back to FP32),
        # while self.score weight can be FP16 or FP32 depending on the model training precision.
        # To avoid precision mismatch,
        # we convert pooled_hidden_states to the same precision as self.score weight.
        if self.has_float16_module_wrapper():
            float16_module = self.module.module
            pooled_hidden_states = fp32_to_float16(pooled_hidden_states, float16_module.float16_convertor)
            need_to_convert_back = True
        else:
            need_to_convert_back = False

        # Assume no bias
        pooled_logits = self.score(pooled_hidden_states)[0]
        pooled_logits = pooled_logits / self.config.temperature
        if need_to_convert_back:
            pooled_logits = float16_to_fp32(pooled_logits)
        return pooled_logits

    @property
    def score(self):
        """Get the score module from the model."""
        if hasattr(self.module, 'score'):
            return self.module.score
        if hasattr(self.module.module, 'score'):
            return self.module.module.score
        assert hasattr(self.module.module.module, 'score'), "Score module not found"
        return self.module.module.module.score

    def has_float16_module_wrapper(self):
        """Check if the model has a float16 module wrapper."""
        if isinstance(self.module, DDP) and isinstance(self.module.module, Float16Module):
            return True
        return False

    @property
    def training_loss_reduction(self):
        """Get the training loss reduction module."""
        if not self._training_loss_reduction:
            self._training_loss_reduction = ReRankerLoss(
                validation_step=False,
                num_hard_negatives=self.config.num_hard_negatives,
                label_smoothing=self.config.label_smoothing,
            )
        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self):
        """Get the validation loss reduction module."""
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = ReRankerLoss(
                validation_step=True,
                num_hard_negatives=self.config.num_hard_negatives,
                label_smoothing=self.config.label_smoothing,
            )
        return self._validation_loss_reduction


@io.model_importer(ReRankerModel, "hf")
class ReRankerImporter(io.ModelConnector["AutoModelForSequenceClassification", ReRankerModel]):
    """HF Importer for Reranker Model"""

    def init(self) -> ReRankerModel:
        return ReRankerModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, trust_remote_code: bool | None = None) -> Path:
        """Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved
            trust_remote_code: Whether remote code execution should be trusted for a given HF path

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import AutoModelForSequenceClassification

        target = self.init()
        trainer = self.nemo_setup(target)
        self.trust_remote_code = trust_remote_code
        source = AutoModelForSequenceClassification.from_pretrained(
            str(self),
            torch_dtype='auto',
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=str(self),
            ),
        )

        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)
        return output_path

    @property
    def config(self) -> ReRankerBaseConfig:
        """Create a NeMo ReRankerBaseConfig from the HF model config."""
        from transformers import AutoConfig

        source = AutoConfig.from_pretrained(
            str(self),
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=str(self),
            ),
        )
        return Llama32Reranker1BConfig(
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            num_layers=source.num_hidden_layers,
        )

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """Get the tokenizer for the HF model.

        Returns:
            AutoTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    def convert_state(self, source: "AutoModelForSequenceClassification", target: ReRankerModel) -> None:
        """Convert the state of the source model to the target model."""
        target_connector = target.config.importer_cls()
        target = target_connector.convert_state(source, target)
        assert (
            target.module.score.weight.dtype == source.score.weight.dtype
        ), f"Score weight dtype mismatch: {target.score.weight.dtype} != {source.score.weight.dtype}"
        with torch.no_grad():
            try:
                target.module.score.weight.copy_(source.score.weight)
            except Exception:
                logging.warning(
                    "Failed to copy score weight. This is expected if you are trying to "
                    "convert model without score weights to NeMo."
                )
                logging.info("init the score weight...")
                target.config.init_method(target.module.score.weight)

        return target


@io.model_exporter(ReRankerModel, "hf")
class ReRankerExporter(io.ModelConnector[ReRankerModel, "AutoModelForSequenceClassification"]):
    """Exporter for converting NeMo Llama models to Hugging Face format.

    This class handles the conversion of NeMo's ReRankerModel to Hugging Face's
    AutoModelForSequenceClassification format, including weight mapping and configuration translation.
    """

    def init(self, dtype=torch.bfloat16) -> "AutoModelForSequenceClassification":
        """Initialize a HF AutoModelForSequenceClassification instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            LlamaBidirectionalForSequenceClassification: Initialized HF Llama Bidirection reranker model
        """
        from transformers.modeling_utils import no_init_weights

        from nemo.collections.llm.gpt.model.hf_llama_embedding import LlamaBidirectionalForSequenceClassification

        with no_init_weights():
            return LlamaBidirectionalForSequenceClassification._from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path, trust_remote_code: bool | None = None) -> Path:
        """Apply the conversion from NeMo to HF format."""
        self.trust_remote_code = trust_remote_code
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
        """Create a NeMo LlamaBidirectionalConfig from the HF model config."""
        source: LlamaConfig = io.load_context(str(self), subpath="model.config")

        from nemo.collections.llm.gpt.model.hf_llama_embedding import (
            LlamaBidirectionalConfig,
            LlamaBidirectionalForSequenceClassification,
        )

        LlamaBidirectionalConfig.register_for_auto_class("AutoConfig")
        LlamaBidirectionalForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
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
            num_labels=1,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
            pad_token_id=self.tokenizer.eos_id,
            temperature=source.temperature,
            rope_scaling={
                "factor": source.scale_factor,
                "high_freq_factor": source.high_freq_factor,
                "low_freq_factor": source.low_freq_factor,
                "original_max_position_embeddings": source.old_context_len,
                "rope_type": "llama3",
            },
        )

    def convert_state(self, source, target):
        """Convert NeMo State dict to HF."""

        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "score.weight": "score.weight",
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


class ReRankerLoss(MegatronLossReduction):
    """Loss function for reranking models that learns to score passages by relevance.

    This loss function implements a cross-entropy based approach for learning to rank passages.
    For each example, it takes a positive passage and multiple hard negative passages,
    and learns to assign higher scores to the positive passage compared to the negatives.

    The loss is computed by treating the scoring task as a classification problem where
    the positive passage should be ranked first among all passages (positive + negatives).
    Cross-entropy loss is used to learn this ranking behavior.

    Args:
        validation_step (bool, optional): Whether this is being used in validation. Defaults to False.
        val_drop_last (bool, optional): Whether to drop the last batch in validation. Defaults to True.
        num_hard_negatives (int, optional): Number of hard negative passages per positive passage. Defaults to 1.
        label_smoothing (float, optional): Label smoothing factor for cross-entropy loss. Defaults to 0.0.

    Note:
        - The input logits should be organized such that for each example, the first score
          corresponds to the positive passage, followed by scores for hard negative passages.
        - The loss assumes all examples have the same number of passages (1 positive + num_hard_negatives).
        - Context parallelism (CP) is not currently supported.
    """

    def __init__(
        self,
        validation_step: bool = False,
        val_drop_last: bool = True,
        num_hard_negatives: int = 1,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last
        self.num_hard_negatives = num_hard_negatives
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from megatron.core import parallel_state

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size != 1:
            raise NotImplementedError(f'CP is not supported for {self.__class__} yet.')

        num_tensors_per_example = 1 + self.num_hard_negatives  #  1 pos, num_hard_negatives negs
        batch_size = forward_out.shape[0] // num_tensors_per_example

        logits = forward_out.view(-1, num_tensors_per_example)
        # Zero labels because we are not trying to predict specific classes.
        # instead, we are learning a scoring function that will help us rank passages in order of relevance.
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        ce_loss = self.cross_entropy_loss(logits, labels)

        reduced_loss = average_losses_across_data_parallel_group([ce_loss])
        return ce_loss, {"avg": reduced_loss}

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """Taken from: https://github.com/NVIDIA/NeMo/blob/main
        /nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L535-L552 ."""
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                # legacy behavior, average over the number of microbatches
                avg = [x["avg"] for x in losses_reduced_per_micro_batch]
                loss = torch.cat(avg).mean()
                return loss

            from megatron.core import parallel_state

            loss_sum_and_ub_size = [
                x["loss_sum_and_ub_size"] for x in losses_reduced_per_micro_batch if x["loss_sum_and_ub_size"][1] > 0
            ]
            loss = (
                torch.vstack(loss_sum_and_ub_size).sum(dim=0)
                if len(loss_sum_and_ub_size) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            torch.distributed.all_reduce(
                loss,
                group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            # average over the total number of tokens across the global batch.
            loss = loss[0] / loss[1]
            return loss

        return torch.tensor(0.0, device=torch.cuda.current_device())


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    from megatron.core import parallel_state

    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=parallel_state.get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(
        group=parallel_state.get_data_parallel_group()
    )

    return averaged_losses
