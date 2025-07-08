# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: skip-file

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.nlp.data.zero_shot_intent_recognition.zero_shot_intent_dataset import (
    ZeroShotIntentDataset,
    ZeroShotIntentInferenceDataset,
    calc_class_weights_from_dataloader,
)

try:
    from nemo.collections.nlp.models import TextClassificationModel
except ImportError:
    from abc import ABC

    TextClassificationModel = ABC
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

__all__ = ['ZeroShotIntentModel']


class ZeroShotIntentModel(TextClassificationModel):
    """TextClassificationModel to be trained on two- or three-class textual entailment data, to be used for zero shot intent recognition."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

    def _setup_dataloader_from_config(self, cfg: DictConfig) -> 'torch.utils.data.DataLoader':
        data_dir = self._cfg.dataset.data_dir
        file_name = cfg.file_name
        input_file = os.path.join(data_dir, file_name)
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"File {input_file} not found. Please check file paths and file names in the config."
            )

        dataset = ZeroShotIntentDataset(
            file_path=input_file,
            tokenizer=self.tokenizer,
            max_seq_length=self._cfg.dataset.max_seq_length,
            sent1_col=self._cfg.dataset.sentence_1_column,
            sent2_col=self._cfg.dataset.sentence_2_column,
            label_col=self._cfg.dataset.label_column,
            num_classes=self.cfg.dataset.num_classes,
            use_cache=self._cfg.dataset.use_cache,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file_name:
            logging.info(
                "Dataloader config or file_name for the training set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

        # calculate the class weights to be used in the loss function
        if self.cfg.dataset.class_balancing == 'weighted_loss':
            self.class_weights = calc_class_weights_from_dataloader(
                self._train_dl, self.cfg.dataset.num_classes, self.cfg.dataset.data_dir
            )
        else:
            self.class_weights = None
        # we need to create/update the loss module by using the weights calculated from the training data
        self.create_loss_module()

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file_name:
            logging.info(
                "Dataloader config or file_path for the validation data set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or not test_data_config.file_name:
            logging.info(
                "Dataloader config or file_path for the test data set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_infer_dataloader(
        self,
        queries: List[str],
        candidate_labels: List[str],
        hypothesis_template=str,
        batch_size=1,
        max_seq_length: int = -1,
    ) -> 'torch.utils.data.DataLoader':
        """
        Setup method for inference data loader. Here the premise-hypothesis pairs are made from queries and candidate labels.

        Args:
            queries: the queries to classify
            candidate_labels: strings to be used as labels
            hypothesis_template: the template used to turn each label into an NLI-style hypothesis. Must include a {}
                or similar syntax for the candidate label to be inserted.
            batch_size: batch size to use during inference
            max_seq_length: maximum length of queries, default is -1 for no limit
        Returns:
            A pytorch DataLoader.
        """
        dataset = ZeroShotIntentInferenceDataset(
            queries=queries,
            candidate_labels=candidate_labels,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length,
            hypothesis_template=hypothesis_template,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

    def predict(
        self,
        queries: Union[str, List[str]],
        candidate_labels: Union[str, List[str]],
        hypothesis_template='This example is {}.',
        batch_size=1,
        multi_label=True,
        entailment_idx=1,
        contradiction_idx=0,
    ) -> List[Dict]:
        """
        Given a list of queries and a list of candidate labels, return a ranked list of labels and scores for each query.

        Example usage:
            queries = ["I'd like a veggie burger, fries, and a coke", "Turn off the lights in the living room",]
            candidate_labels = ["Food order", "Change lighting"]
            model.predict(queries, candidate_labels)

        Example output:
            [{'sentence': "I'd like a veggie burger, fries, and a coke",
              'labels': ['Food order', 'Change lighting'],
              'scores': [0.8557153344154358, 0.12036784738302231]},
             {'sentence': 'Turn off the lights in the living room',
              'labels': ['Change lighting', 'Food order'],
              'scores': [0.8506497144699097, 0.06594637036323547]}]


        Args:
            queries: the query or list of queries to classify
            candidate_labels: string or list of strings to be used as labels
            hypothesis_template: the template used to turn each label into an NLI-style hypothesis. Must include a {}
            or similar syntax for the candidate label to be inserted.
            batch_size: the batch size to use for inference.
            multi_label: whether or not multiple candidate labels can be true. If False, the scores are normalized
            such that all class probabilities sum to 1. If True, the labels are
            considered independent and probabilities are normalized for each candidate by doing a softmax of
            the entailment score vs. the contradiction score.
            entailment_idx: the index of the "entailment" class in the trained model; models trained on MNLI
             using NeMo's glue_benchmark.py or zero_shot_intent_model.py use an index of 1 by default.
            contradiction_idx: the index of the "contradiction" class in the trained model; models trained on MNLI
             using NeMo's glue_benchmark.py or zero_shot_intent_model.py use an index of 0 by default.

        Returns:
            list of dictionaries; one dict per input query. Each dict has keys "sentence", "labels", "scores".
            labels and scores are parallel lists (with each score corresponding to the label at the same index),
                 sorted from highest to lowest score.

        """
        if not queries:
            raise ValueError("No queries were passed for classification!")
        if not candidate_labels:
            raise ValueError("No candidate labels were provided!")

        queries = [queries] if isinstance(queries, str) else queries
        candidate_labels = [candidate_labels] if isinstance(candidate_labels, str) else candidate_labels

        if len(candidate_labels) == 1:
            multi_label = True

        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            infer_datalayer = self._setup_infer_dataloader(
                queries,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                batch_size=batch_size,
                max_seq_length=self._cfg.dataset.max_seq_length,
            )

            all_batch_logits = []
            for batch in infer_datalayer:
                input_ids, input_type_ids, input_mask, _ = batch

                logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )
                all_batch_logits.append(logits.detach().cpu().numpy())

            all_logits = np.concatenate(all_batch_logits)
            outputs = all_logits.reshape((len(queries), len(candidate_labels), -1))

            if not multi_label:
                # softmax the "entailment" logits over all candidate labels
                entail_logits = outputs[..., entailment_idx]
                scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
            else:
                # softmax over the entailment vs. contradiction dim for each label independently
                entail_contr_logits = outputs[..., [contradiction_idx, entailment_idx]]
                scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
                scores = scores[..., 1]

            result = []
            for i in range(len(queries)):
                sorted_idxs = list(reversed(scores[i].argsort()))
                result.append(
                    {
                        "sentence": queries[i],
                        "labels": [candidate_labels[j] for j in sorted_idxs],
                        "scores": scores[i][sorted_idxs].tolist(),
                    }
                )

        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return result

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained models which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="zeroshotintent_en_bert_base_uncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/zeroshotintent_en_bert_base_uncased/versions/1.4.1/files/zeroshotintent_en_bert_base_uncased.nemo",
                description="ZeroShotIntentModel trained by fine tuning BERT-base-uncased on the MNLI (Multi-Genre Natural Language Inference) dataset, which achieves an accuracy of 84.9% and 84.8% on the matched and mismatched dev sets, respectively.",
            )
        )
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="zeroshotintent_en_megatron_uncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/zeroshotintent_en_megatron_uncased/versions/1.4.1/files/zeroshotintent_en_megatron_uncased.nemo",
                description="ZeroShotIntentModel trained by fine tuning Megatron-BERT-345m=M-uncased on the MNLI (Multi-Genre Natural Language Inference) dataset, which achieves an accuracy of 90.0% and 89.9% on the matched and mismatched dev sets, respectively",
            )
        )
        return result
