# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Optional

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.nlp.data import BertInformationRetrievalDataset
from nemo.collections.nlp.models.information_retrieval.base_ir_model import BaseIRModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import ChannelType, LogitsType, MaskType, NeuralType

__all__ = ["BertDPRModel"]


class BertDPRModel(BaseIRModel):
    """
    Information retrieval model which encodes query and passage separately
    with two different BERT encoders and computes their similarity score
    as a dot-product between corresponding [CLS] token representations.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "q_input_ids": NeuralType(("B", "T"), ChannelType()),
            "q_attention_mask": NeuralType(("B", "T"), MaskType()),
            "q_token_type_ids": NeuralType(("B", "T"), ChannelType()),
            "p_input_ids": NeuralType(("B", "T"), ChannelType()),
            "p_attention_mask": NeuralType(("B", "T"), MaskType()),
            "p_token_type_ids": NeuralType(("B", "T"), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(("B", "D"), LogitsType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        model_name = cfg.language_model.pretrained_model_name
        self.tokenizer = get_tokenizer(tokenizer_name=model_name)

        super().__init__(cfg=cfg, trainer=trainer)

        self.q_encoder = self.get_lm_model_with_padded_embedding(cfg)
        self.p_encoder = self.get_lm_model_with_padded_embedding(cfg)
        self.loss = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id)

    @typecheck()
    def forward(
        self,
        q_input_ids,
        q_token_type_ids,
        q_attention_mask,
        p_input_ids,
        p_token_type_ids,
        p_attention_mask,
    ):

        q_vectors = self.q_encoder(
            input_ids=q_input_ids,
            token_type_ids=q_token_type_ids,
            attention_mask=q_attention_mask,
        )
        q_vectors = q_vectors[:, 0]
        batch_size, hidden_size = q_vectors.size()

        p_vectors = self.p_encoder(
            input_ids=p_input_ids,
            token_type_ids=p_token_type_ids,
            attention_mask=p_attention_mask,
        )
        num_passages = p_vectors.shape[0] // batch_size
        p_vectors = p_vectors[:, 0].view(-1, num_passages, hidden_size)
        p_positives, p_negatives = p_vectors[:, 0], p_vectors[:, 1:]
        scores = torch.cat(
            (
                torch.matmul(q_vectors, p_positives.T),
                torch.einsum("ij,ipj->ip", q_vectors, p_negatives),
            ),
            dim=1,
        )

        return scores

    def compute_scores_and_loss(self, inputs):
        (
            q_input_ids,
            q_input_mask,
            q_input_type_ids,
            p_input_ids,
            p_input_mask,
            p_input_type_ids,
        ) = inputs
        batch_size, num_passages, p_seq_length = p_input_ids.size()
        q_seq_length = q_input_ids.size()[-1]

        scores = self(
            q_input_ids=q_input_ids.view(-1, q_seq_length),
            q_token_type_ids=q_input_type_ids.view(-1, q_seq_length),
            q_attention_mask=q_input_mask.view(-1, q_seq_length),
            p_input_ids=p_input_ids.view(-1, p_seq_length),
            p_token_type_ids=p_input_type_ids.view(-1, p_seq_length),
            p_attention_mask=p_input_mask.view(-1, p_seq_length),
        ).view(batch_size, 1, batch_size + num_passages - 1)
        normalized_scores = torch.log_softmax(scores, dim=-1)

        labels = torch.arange(batch_size)[:, None].long().to(normalized_scores.device)
        loss = self.loss(
            log_probs=normalized_scores,
            labels=labels,
            output_mask=torch.ones_like(labels),
        )

        scores = scores[:, 0]
        scores = torch.cat(
            (torch.diag(scores)[:, None], scores[:, batch_size:]),
            dim=1,
        )

        return scores, loss

    def _setup_dataloader_from_config(self, cfg: DictConfig):

        dataset = BertInformationRetrievalDataset(
            tokenizer=self.tokenizer,
            passages=cfg.passages,
            queries=cfg.queries,
            query_to_passages=cfg.query_to_passages,
            num_negatives=cfg.num_negatives,
            psg_cache_format=cfg.get("psg_cache_format", "pkl"),
            max_query_length=cfg.get("max_query_length", 31),
            max_passage_length=cfg.get("max_passage_length", 190),
            preprocess_fn="preprocess_dpr",
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
