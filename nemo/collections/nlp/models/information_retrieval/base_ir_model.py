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

import math
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.nlp.data import BertInformationRetrievalDataset
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.core.classes.common import typecheck

__all__ = ['BaseIRModel']


class BaseIRModel(NLPModel):
    """
    Base class for information retrieval models.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        self.setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

    @typecheck()
    def forward(self, *args):
        pass

    def compute_scores_and_loss(self, inputs):
        pass

    @staticmethod
    def get_lm_model_with_padded_embedding(cfg: DictConfig):
        """
        Function which ensures that vocabulary size is divisivble by 8
        for faster mixed precision training.
        """
        model = get_lm_model(
            config_file=cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            vocab_file=cfg.tokenizer.vocab_file,
            trainer=trainer,
            cfg=cfg,
        )
        vocab_size, hidden_size = model.config.vocab_size, model.config.hidden_size
        tokens_to_add = 8 * math.ceil(vocab_size / 8) - vocab_size
        zeros = torch.zeros((tokens_to_add, hidden_size))
        model.embeddings.word_embeddings.weight.data = torch.cat((model.embeddings.word_embeddings.weight.data, zeros))
        return model

    @staticmethod
    def calculate_mean_reciprocal_rank(query2passages, query2rel):
        """
        Helper function which calculates mean reciprocal rank.
        Args:
            query2passages: dict which contains passage ids and corresponding
                scores for each query
            query2rel: dict which contains ids of relevant passages for each query
        """
        reciprocal_ranks = []

        for query in query2passages:
            indices = np.argsort(query2passages[query]["scores"])[::-1]
            sorted_psgs = query2passages[query]["psg_ids"][indices]

            reciprocal_ranks.append(0)
            for i, psg_id in enumerate(sorted_psgs):
                if psg_id in query2rel[query]:
                    reciprocal_ranks[-1] = 1 / (i + 1)
                    break
        return np.mean(reciprocal_ranks)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        scores, train_loss = self.compute_scores_and_loss(batch[:-2])
        tensorboard_logs = {"train_loss": train_loss, "lr": self._optimizer.param_groups[0]["lr"]}
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        scores, val_loss = self.compute_scores_and_loss(batch[:-2])
        query_ids, passage_ids = batch[-2:]
        data_for_val = {
            "val_loss": val_loss,
            "scores": scores,
            "query_ids": query_ids,
            "passage_ids": passage_ids,
        }
        self.validation_step_outputs.append(data_for_val)
        return data_for_val

    def on_validation_epoch_end(self):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        query_ids = torch.cat([x["query_ids"] for x in self.validation_step_outputs])
        passage_ids = torch.cat([x["passage_ids"] for x in self.validation_step_outputs])
        scores = torch.cat([x["scores"] for x in self.validation_step_outputs])

        all_query_ids, all_passage_ids, all_scores = [], [], []
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                all_query_ids.append(torch.empty_like(query_ids))
                all_passage_ids.append(torch.empty_like(passage_ids))
                all_scores.append(torch.empty_like(scores))
            torch.distributed.all_gather(all_query_ids, query_ids)
            torch.distributed.all_gather(all_passage_ids, passage_ids)
            torch.distributed.all_gather(all_scores, scores)
        else:
            all_query_ids.append(query_ids)
            all_passage_ids.append(passage_ids)
            all_scores.append(scores)

        val_mrr = 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            query2passages, query2rels = {}, {}
            processed_queries = set()

            for i in range(len(all_query_ids)):

                query_ids = all_query_ids[i].detach().cpu().numpy()
                passage_ids = all_passage_ids[i].detach().cpu().numpy()
                scores = all_scores[i].detach().cpu().numpy()

                for j, query_id in enumerate(query_ids):

                    if query_id not in processed_queries:
                        processed_queries.add(query_id)
                        query2passages[query_id] = {
                            "psg_ids": passage_ids[j],
                            "scores": scores[j],
                        }
                        query2rels[query_id] = [passage_ids[j][0]]
                    else:
                        query2passages[query_id]["psg_ids"] = np.concatenate(
                            (query2passages[query_id]["psg_ids"], passage_ids[j][1:])
                        )
                        query2passages[query_id]["scores"] = np.concatenate(
                            (query2passages[query_id]["scores"], scores[j][1:])
                        )

            val_mrr = self.calculate_mean_reciprocal_rank(query2passages, query2rels)

        val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()  # free memory
        tensorboard_logs = {
            "val_mrr": val_mrr,
            "val_loss": val_loss,
        }

        return {"log": tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

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
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
