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
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from torch import Tensor, nn

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.llm.bert.loss import BERTInBatchExclusiveHardNegativesRankingLoss
from nemo.collections.llm.bert.model import BertConfig, BertModel
from nemo.collections.llm.bert.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params
from nemo.collections.llm.bert.model.bert import HuggingFaceBertImporter
from nemo.lightning import io
from nemo.lightning.pytorch.optim import OptimizerModule


def bert_embedding_data_step(dataloder_iter) -> Dict[str, torch.Tensor]:
    """Setup BERT dataloader batch."""
    batch = next(dataloder_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    required_keys.add("token_type_ids")

    if parallel_state.is_pipeline_first_stage():
        required_keys.add("input_ids")

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def bert_embedding_forward_step(model: L.LightningModule, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    This subsets the batch keys to the ones actually used by forward pass of the model,
    and then calls the model's forward pass. if "cu_seqsens" are defined in the batch,
    then the packed sequence parameters are also passed to the model for forward pass efficiency.
    """
    forward_args = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }

    if model.config.num_tokentypes != 0:
        forward_args["tokentype_ids"] = batch["token_type_ids"]

    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)

    return model(**forward_args)


@dataclass
class BertEmbeddingConfig(BertConfig):
    """Bert Embedding Config"""

    bert_type: Literal["huggingface", "megatron"] = 'huggingface'
    ce_loss_scale: float = 20
    label_smoothing: float = 0.0
    add_lm_head: bool = False
    bert_binary_head: bool = False
    num_hard_negatives: int = 1
    num_tokentypes: int = 2
    global_in_batch_negatives: bool = True
    backprop_type: Literal["local", "global"] = 'local'
    forward_step_fn: Callable = bert_embedding_forward_step
    data_step_fn: Callable = bert_embedding_data_step


@dataclass
class BertEmbeddingLargeConfig(BertEmbeddingConfig):
    """Bert Embedding model follows Bert-large architecture."""

    num_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16


@dataclass
class BertEmbeddingMiniConfig(BertEmbeddingConfig):
    """Bert Embedding model follows Bert-mini (384 hidden size) architecture."""

    num_layers: int = 6
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_attention_heads: int = 12


class BertEmbeddingHead(nn.Module):
    """Performs mean pooling on the token embeddings."""

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode_mean_tokens: bool = True,
    ):
        super(BertEmbeddingHead, self).__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_mean_tokens",
        ]
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor):
        """Forward function for embedding head. Performs mean pooling."""
        token_embeddings = token_embeddings.permute(1, 0, 2)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        output_vector = sum_embeddings / sum_mask
        output_vector = F.normalize(output_vector, p=2, dim=1)

        return output_vector


class BertEmbeddingModel(BertModel):
    """Bert Lightning Module"""

    def __init__(
        self,
        config: BertConfig,
        # TODO: Add transformer_layer_spec when we update mcore
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config, optim, tokenizer, model_transform)

    def configure_model(self) -> None:
        """Setup the BERT Model based on config definition."""
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)
            self.embedding_head = BertEmbeddingHead(
                word_embedding_dimension=self.config.hidden_size,
                pooling_mode_mean_tokens=True,
            )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Call the forward method of the underlying model, and return whatever it outputs."""
        assert "attention_mask" in kwargs, "attention mask is required for BERT Embedding Model."
        output_tensor = self.module(
            hidden_states_only=True, *args, **kwargs
        )  # for now just pass through to the underlying model
        embeddings_out = self.embedding_head(output_tensor, kwargs["attention_mask"])
        return embeddings_out

    @property
    def training_loss_reduction(self) -> BERTInBatchExclusiveHardNegativesRankingLoss:  # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = BERTInBatchExclusiveHardNegativesRankingLoss(
                validation_step=False,
                num_hard_negatives=self.config.num_hard_negatives,
                scale=self.config.ce_loss_scale,
                label_smoothing=self.config.label_smoothing,
                global_in_batch_negatives=self.config.global_in_batch_negatives,
                backprop_type=self.config.backprop_type,
            )

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> BERTInBatchExclusiveHardNegativesRankingLoss:  # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = BERTInBatchExclusiveHardNegativesRankingLoss(
                validation_step=True,
                num_hard_negatives=self.config.num_hard_negatives,
                scale=self.config.ce_loss_scale,
                label_smoothing=self.config.label_smoothing,
            )

        return self._validation_loss_reduction


@io.model_importer(BertEmbeddingModel, "hf")
class BertEmbeddingImporter(HuggingFaceBertImporter):
    """
    Importer for BertEmbedding Model.
    HuggingFace uses same model for Bert Embedding model and Bert model, thus the connector is identical.
    """

    def __init__(self, *args, **kwargs):
        if sys.version_info > (3, 11):
            # In Python versions <= 3.11, *Path classes don’t have a __init__ method,
            # and do all their initialization in __new__/ helper methods.
            # Only need to call super().__init__ if version > 3.11
            super().__init__(*args)
        self.type = 'model'

    def init(self) -> BertEmbeddingModel:
        return BertEmbeddingModel(self.config, tokenizer=self.tokenizer)
