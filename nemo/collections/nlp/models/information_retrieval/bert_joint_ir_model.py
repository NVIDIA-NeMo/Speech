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
from nemo.collections.nlp.models.information_retrieval.base_ir_model import BaseIRModel
from nemo.collections.nlp.modules.common import SequenceRegression
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import NeuralType

__all__ = ["BertJointIRModel"]


class BertJointIRModel(BaseIRModel):
    """
    Information retrieval model which jointly encodes both query and passage
    and passes them to BERT encoder followed by a fully-connected layer for
    similarity score prediction.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.sim_score_regressor.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        model_name = cfg.language_model.pretrained_model_name
        self.tokenizer = get_tokenizer(tokenizer_name=model_name)

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = self.get_lm_model_with_padded_embedding(cfg)
        hidden_size = self.bert_model.config.hidden_size
        self.sim_score_regressor = SequenceRegression(
            hidden_size=hidden_size,
            num_layers=1,
            dropout=cfg.language_model.sim_score_dropout,
        )
        self.loss = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id)

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids):

        hidden_states = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        scores = self.sim_score_regressor(hidden_states=hidden_states)

        return scores

    def compute_scores_and_loss(self, inputs):
        input_ids, input_mask, input_type_ids = inputs
        batch_size, num_passages, seq_length = input_ids.size()

        unnormalized_scores = self(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=input_mask.view(-1, seq_length),
            token_type_ids=input_type_ids.view(-1, seq_length),
        ).view(batch_size, 1, num_passages)
        scores = torch.log_softmax(unnormalized_scores, dim=-1)

        labels = torch.zeros_like(input_ids[:, :1, 0])
        loss = self.loss(log_probs=scores, labels=labels, output_mask=torch.ones_like(labels))

        return unnormalized_scores[:, 0], loss
