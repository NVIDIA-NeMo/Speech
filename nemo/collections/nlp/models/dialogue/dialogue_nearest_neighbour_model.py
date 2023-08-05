# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import AutoModel

from nemo.collections.nlp.data.dialogue import DialogueSGDDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.assistant_data_processor import DialogueAssistantDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.design_data_processor import DialogueDesignDataProcessor
from nemo.collections.nlp.data.dialogue.dataset.dialogue_nearest_neighbour_dataset import (
    DialogueNearestNeighbourDataset,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.metrics.dialogue_metrics import DialogueGenerationMetrics
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

__all__ = ['DialogueNearestNeighbourModel']


class DialogueNearestNeighbourModel(NLPModel):
    """Dialogue Nearest Neighbour Model identifies the intent of an utterance using the cosine similarity between sentence embeddings of the utterance and various label descriptions """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.cfg = cfg
        super().__init__(cfg=cfg, trainer=trainer)
        if self.cfg.library == "huggingface":
            self.language_model = AutoModel.from_pretrained(self.cfg.language_model.pretrained_model_name)

    def _setup_dataloader_from_config(self, cfg: DictConfig, dataset_split) -> 'torch.utils.data.DataLoader':
        if self._cfg.dataset.task == "zero_shot":
            self.data_processor = DialogueAssistantDataProcessor(
                self.cfg.data_dir, self.tokenizer, cfg=self.cfg.dataset
            )
        elif self._cfg.dataset.task == "design":
            self.data_processor = DialogueDesignDataProcessor(
                data_dir=self._cfg.dataset.data_dir, tokenizer=self.tokenizer, cfg=self._cfg.dataset
            )
        elif self._cfg.dataset.task == 'sgd':
            self.data_processor = DialogueSGDDataProcessor(
                data_dir=self._cfg.dataset.data_dir,
                dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
                tokenizer=self.tokenizer,
                cfg=self._cfg.dataset,
            )
        else:
            raise ValueError("Only zero_shot, design and sgd supported for Zero Shot Intent Model")

        dataset = DialogueNearestNeighbourDataset(
            dataset_split,
            self.data_processor,
            self.tokenizer,
            self.cfg.dataset,  # this is the model.dataset cfg, which is diff from train_ds cfg etc
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

    def forward(self, input_ids, attention_mask):
        if self.cfg.library == 'huggingface':
            output = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        return output

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        loss = self.validation_step(batch, batch_idx, mode='test')
        self.test_step_outputs.append(loss)
        return loss

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def validation_step(self, batch, batch_idx, mode='val'):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_mask, labels = batch
        preds = []
        gts = []
        inputs = []
        for i in range(input_ids.size(0)):
            output = self.forward(input_ids=input_ids[i], attention_mask=input_mask[i])
            sentence_embeddings = DialogueNearestNeighbourModel.mean_pooling(output, input_mask[i])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            cos_sim = F.cosine_similarity(sentence_embeddings[:1, :], sentence_embeddings[1:, :])
            pred = torch.argmax(cos_sim).item() + 1
            gt = torch.argmax(labels[i][1:]).item() + 1

            preds.append(input_ids[i, pred])
            gts.append(input_ids[i, gt])
            inputs.append(input_ids[i, 0])

        loss = {'preds': torch.stack(preds), 'labels': torch.stack(gts), 'inputs': torch.stack(inputs)}
        self.validation_step_outputs.append(loss)
        return loss

    def multi_test_epoch_end(self, outputs, dataloader_idx):
        return self.on_validation_epoch_end()

    def on_validation_epoch_end(self):
        """
        Get metrics based on the candidate label with the highest predicted likelihood and the ground truth label for intent
        """
        prefix = "test" if self.trainer.testing else "val"
        if prefix == "val":
            outputs = self.validation_step_outputs
        else:
            outputs = self.test_step_outputs
        output_preds = torch.cat([output['preds'] for output in outputs], dim=0)
        output_labels = torch.cat([output['labels'] for output in outputs], dim=0)
        inputs = torch.cat([output['inputs'] for output in outputs], dim=0)

        decoded_preds = self.tokenizer.tokenizer.batch_decode(output_preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.tokenizer.batch_decode(output_labels, skip_special_tokens=True)
        decoded_inputs = self.tokenizer.tokenizer.batch_decode(inputs, skip_special_tokens=True)

        prompt_len = len(self.cfg.dataset.prompt_template.strip())
        predicted_labels = [i[prompt_len:].strip() for i in decoded_preds]
        ground_truth_labels = [i[prompt_len:].strip() for i in decoded_labels]

        os.makedirs(self.cfg.dataset.dialogues_example_dir, exist_ok=True)
        filename = os.path.join(self.cfg.dataset.dialogues_example_dir, "test_predictions.jsonl")

        DialogueGenerationMetrics.save_predictions(
            filename, predicted_labels, ground_truth_labels, decoded_inputs,
        )

        label_to_ids = {label: idx for idx, label in enumerate(list(set(predicted_labels + ground_truth_labels)))}
        self.classification_report = ClassificationReport(
            num_classes=len(label_to_ids), mode='micro', label_ids=label_to_ids, dist_sync_on_step=True
        ).to(output_preds[0].device)

        predicted_label_ids = torch.tensor([label_to_ids[label] for label in predicted_labels]).to(
            output_preds[0].device
        )
        ground_truth_label_ids = torch.tensor([label_to_ids[label] for label in ground_truth_labels]).to(
            output_preds[0].device
        )

        tp, fn, fp, _ = self.classification_report(predicted_label_ids, ground_truth_label_ids)

        precision, recall, f1, report = self.classification_report.compute()
        label_acc = np.mean([int(predicted_labels[i] == ground_truth_labels[i]) for i in range(len(predicted_labels))])

        logging.info(report)

        self.log('unified_precision', precision)
        self.log('unified_f1', f1)
        self.log('unified_recall', recall)
        self.log('unfied_accuracy', label_acc * 100)

        self.classification_report.reset()
        self.validation_step_outputs.clear() if prefix == 'val' else self.test_step_outputs.clear()

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config:
            logging.info(
                f"Dataloader config or file_name for the training set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(train_data_config, "train")

        # self.create_loss_module()

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config:
            logging.info(
                f"Dataloader config or file_path for the validation data set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(val_data_config, "dev")

    def setup_multiple_test_data(self, test_data_config: Optional[DictConfig]):
        self.setup_test_data(test_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config:
            logging.info(
                f"Dataloader config or file_path for the test data set is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(test_data_config, "test")

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained models which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result
