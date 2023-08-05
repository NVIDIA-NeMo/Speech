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
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from nemo.collections.common.losses import AggregatorLoss, BCEWithLogitsLoss, CrossEntropyLoss
from nemo.collections.nlp.data.intent_slot_classification import (
    MultiLabelIntentSlotClassificationDataset,
    MultiLabelIntentSlotDataDesc,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport, MultiLabelClassificationReport
from nemo.collections.nlp.models.intent_slot_classification import IntentSlotClassificationModel
from nemo.collections.nlp.modules.common import SequenceTokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging


class MultiLabelIntentSlotClassificationModel(IntentSlotClassificationModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """ 
        Initializes BERT Joint Intent and Slot model.

        Args: 
            cfg: configuration object
            trainer: trainer for Pytorch Lightning
        """
        self.max_seq_length = cfg.language_model.max_seq_length

        # Optimal Threshold
        self.threshold = 0.5
        self.max_f1 = 0

        # Check the presence of data_dir.
        if not cfg.data_dir or not os.path.exists(cfg.data_dir):
            # Set default values of data_desc.
            self._set_defaults_data_desc(cfg)
        else:
            self.data_dir = cfg.data_dir
            # Update configuration of data_desc.
            self._set_data_desc_to_cfg(cfg, cfg.data_dir, cfg.train_ds, cfg.validation_ds)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # Initialize Classifier.
        self._reconfigure_classifier()

    def _set_data_desc_to_cfg(
        self, cfg: DictConfig, data_dir: str, train_ds: DictConfig, validation_ds: DictConfig
    ) -> None:
        """ 
        Creates MultiLabelIntentSlotDataDesc and copies generated values to Configuration object's data descriptor. 
        
        Args: 
            cfg: configuration object
            data_dir: data directory 
            train_ds: training dataset file name
            validation_ds: validation dataset file name

        Returns:
            None
        """
        # Save data from data desc to config - so it can be reused later, e.g. in inference.
        data_desc = MultiLabelIntentSlotDataDesc(data_dir=data_dir, modes=[train_ds.prefix, validation_ds.prefix])
        OmegaConf.set_struct(cfg, False)
        if not hasattr(cfg, "data_desc") or cfg.data_desc is None:
            cfg.data_desc = {}
        # Intents.
        cfg.data_desc.intent_labels = list(data_desc.intents_label_ids.keys())
        cfg.data_desc.intent_label_ids = data_desc.intents_label_ids
        cfg.data_desc.intent_weights = data_desc.intent_weights
        # Slots.
        cfg.data_desc.slot_labels = list(data_desc.slots_label_ids.keys())
        cfg.data_desc.slot_label_ids = data_desc.slots_label_ids
        cfg.data_desc.slot_weights = data_desc.slot_weights

        cfg.data_desc.pad_label = data_desc.pad_label

        # for older(pre - 1.0.0.b3) configs compatibility
        if not hasattr(cfg, "class_labels") or cfg.class_labels is None:
            cfg.class_labels = {}
            cfg.class_labels = OmegaConf.create(
                {"intent_labels_file": "intent_labels.csv", "slot_labels_file": "slot_labels.csv",}
            )

        slot_labels_file = os.path.join(data_dir, cfg.class_labels.slot_labels_file)
        intent_labels_file = os.path.join(data_dir, cfg.class_labels.intent_labels_file)
        self._save_label_ids(data_desc.slots_label_ids, slot_labels_file)
        self._save_label_ids(data_desc.intents_label_ids, intent_labels_file)

        self.register_artifact("class_labels.intent_labels_file", intent_labels_file)
        self.register_artifact("class_labels.slot_labels_file", slot_labels_file)
        OmegaConf.set_struct(cfg, True)

    def _reconfigure_classifier(self) -> None:
        """ Method reconfigures the classifier depending on the settings of model cfg.data_desc """

        self.classifier = SequenceTokenClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_intents=len(self.cfg.data_desc.intent_labels),
            num_slots=len(self.cfg.data_desc.slot_labels),
            dropout=self.cfg.head.fc_dropout,
            num_layers=self.cfg.head.num_output_layers,
            log_softmax=False,
        )

        # define losses
        if self.cfg.class_balancing == "weighted_loss":
            # You may need to increase the number of epochs for convergence when using weighted_loss
            self.intent_loss = BCEWithLogitsLoss(logits_ndim=2, pos_weight=self.cfg.data_desc.intent_weights)
            self.slot_loss = CrossEntropyLoss(logits_ndim=3, weight=self.cfg.data_desc.slot_weights)
        else:
            self.intent_loss = BCEWithLogitsLoss(logits_ndim=2)
            self.slot_loss = CrossEntropyLoss(logits_ndim=3)

        self.total_loss = AggregatorLoss(
            num_inputs=2, weights=[self.cfg.intent_loss_weight, 1.0 - self.cfg.intent_loss_weight],
        )

        # setup to track metrics
        self.intent_classification_report = MultiLabelClassificationReport(
            num_classes=len(self.cfg.data_desc.intent_labels),
            label_ids=self.cfg.data_desc.intent_label_ids,
            dist_sync_on_step=True,
            mode="micro",
        )
        self.slot_classification_report = ClassificationReport(
            num_classes=len(self.cfg.data_desc.slot_labels),
            label_ids=self.cfg.data_desc.slot_label_ids,
            dist_sync_on_step=True,
            mode="micro",
        )

    def validation_step(self, batch, batch_idx) -> None:
        """
        Validation Loop. Pytorch Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.

        Args:
            batch: batches of data from DataLoader
            batch_idx: batch idx from DataLoader

        Returns: 
            None
        """
        (input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, intent_labels, slot_labels,) = batch
        intent_logits, slot_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask,
        )

        # calculate combined loss for intents and slots
        intent_loss = self.intent_loss(logits=intent_logits, labels=intent_labels)
        slot_loss = self.slot_loss(logits=slot_logits, labels=slot_labels, loss_mask=loss_mask)
        val_loss = self.total_loss(loss_1=intent_loss, loss_2=slot_loss)

        intent_probabilities = torch.round(torch.sigmoid(intent_logits))

        self.intent_classification_report.update(intent_probabilities, intent_labels)
        # slots
        subtokens_mask = subtokens_mask > 0.5
        preds = torch.argmax(slot_logits, axis=-1)[subtokens_mask]
        slot_labels = slot_labels[subtokens_mask]
        self.slot_classification_report.update(preds, slot_labels)

        loss = {
            "val_loss": val_loss,
            "intent_tp": self.intent_classification_report.tp,
            "intent_fn": self.intent_classification_report.fn,
            "intent_fp": self.intent_classification_report.fp,
            "slot_tp": self.slot_classification_report.tp,
            "slot_fn": self.slot_classification_report.fn,
            "slot_fp": self.slot_classification_report.fp,
        }
        self.validation_step_outputs.append(loss)
        return loss

    def _setup_dataloader_from_config(self, cfg: DictConfig) -> DataLoader:
        """
        Creates the DataLoader from the configuration object

        Args:
            cfg: configuration object
        
        Returns:
            DataLoader for model's data
        """

        input_file = f"{self.data_dir}/{cfg.prefix}.tsv"
        slot_file = f"{self.data_dir}/{cfg.prefix}_slots.tsv"
        intent_dict_file = self.data_dir + "/dict.intents.csv"

        lines = open(intent_dict_file, "r").readlines()
        lines = [line.strip() for line in lines if line.strip()]
        num_intents = len(lines)

        if not (os.path.exists(input_file) and os.path.exists(slot_file)):
            raise FileNotFoundError(
                f"{input_file} or {slot_file} not found. Please refer to the documentation for the right format \
                 of Intents and Slots files."
            )

        dataset = MultiLabelIntentSlotClassificationDataset(
            input_file=input_file,
            slot_file=slot_file,
            num_intents=num_intents,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            num_samples=cfg.num_samples,
            pad_label=self.cfg.data_desc.pad_label,
            ignore_extra_tokens=self.cfg.ignore_extra_tokens,
            ignore_start_end=self.cfg.ignore_start_end,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            collate_fn=dataset.collate_fn,
        )

    def prediction_probabilities(self, queries: List[str], test_ds: DictConfig) -> npt.NDArray:
        """
        Get prediction probabilities for the queries (intent and slots)

        Args:
            queries: text sequences
            test_ds: Dataset configuration section.

        Returns:
            numpy array of intent probabilities
        """

        probabilities = []

        mode = self.training
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            # Dataset.
            infer_datalayer = self._setup_infer_dataloader(queries, test_ds)

            for batch in infer_datalayer:
                input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask = batch

                intent_logits, slot_logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )

                # predict intents for these examples
                probabilities.append(torch.sigmoid(intent_logits).detach().cpu().numpy())

            probabilities = np.concatenate(probabilities)

        finally:
            # set mode back to its original value
            self.train(mode=mode)

        return probabilities

    def optimize_threshold(self, test_ds: DictConfig, file_name: str) -> None:
        """
        Set the optimal threshold of the model from performance on validation set. This threshold is used to round the 
        logits to 0 or 1. 

        Args:
            test_ds: location of test dataset
            file_name: name of input file to retrieve validation set

        Returns:
            None
        """

        input_file = f"{self.data_dir}/{file_name}.tsv"

        with open(input_file, "r") as f:
            input_lines = f.readlines()[1:]  # Skipping headers at index 0

        dataset = list(input_lines)

        metrics_labels, sentences = [], []

        for input_line in dataset:
            sentence = input_line.strip().split("\t")[0]
            sentences.append(sentence)
            parts = input_line.strip().split("\t")[1:][0]
            parts = list(map(int, parts.split(",")))
            parts = [1 if label in parts else 0 for label in range(len(self.cfg.data_desc.intent_labels))]
            metrics_labels.append(parts)

        # Retrieve class probabilities for each sentence
        intent_probabilities = self.prediction_probabilities(sentences, test_ds)

        metrics_dict = {}
        # Find optimal logits rounding threshold for intents
        for i in np.arange(0.5, 0.96, 0.01):
            predictions = (intent_probabilities >= i).tolist()
            precision = precision_score(metrics_labels, predictions, average='micro')
            recall = recall_score(metrics_labels, predictions, average='micro')
            f1 = f1_score(metrics_labels, predictions, average='micro')
            metrics_dict[i] = [precision, recall, f1]

        max_precision = max(metrics_dict, key=lambda x: metrics_dict[x][0])
        max_recall = max(metrics_dict, key=lambda x: metrics_dict[x][1])
        max_f1_score = max(metrics_dict, key=lambda x: metrics_dict[x][2])

        logging.info(
            f'Best Threshold for F1-Score: {max_f1_score}, [Precision, Recall, F1-Score]: {metrics_dict[max_f1_score]}'
        )
        logging.info(
            f'Best Threshold for Precision: {max_precision}, [Precision, Recall, F1-Score]: {metrics_dict[max_precision]}'
        )
        logging.info(
            f'Best Threshold for Recall: {max_recall}, [Precision, Recall, F1-Score]: {metrics_dict[max_recall]}'
        )

        if metrics_dict[max_f1_score][2] > self.max_f1:
            self.max_f1 = metrics_dict[max_f1_score][2]

            logging.info(f'Setting Threshold to: {max_f1_score}')

            self.threshold = max_f1_score

    def predict_from_examples(
        self, queries: List[str], test_ds: DictConfig, threshold: float = None
    ) -> Tuple[List[List[Tuple[str, float]]], List[str], List[List[int]]]:
        """
        Get prediction for the queries (intent and slots)


        Args:
            queries: text sequences
            test_ds: Dataset configuration section.
            threshold: Threshold for rounding prediction logits
        
        Returns:
            predicted_intents: model intent predictions with their probabilities
                Example:  [[('flight', 0.84)], [('airfare', 0.54), 
                            ('flight', 0.73), ('meal', 0.24)]]
            predicted_slots: model slot predictions
                Example:  ['O B-depart_date.month_name B-depart_date.day_number',
                           'O O B-flight_stop O O O']

            predicted_vector: model intent predictions for each individual query. Binary values within each list 
                indicate whether a class is prediced for the given query (1 for True, 0 for False)
                Example: [[1,0,0,0,0,0], [0,0,1,0,0,0]]
        """
        predicted_intents = []

        if threshold is None:
            threshold = self.threshold
        logging.info(f'Using threshold = {threshold}')

        predicted_slots = []
        predicted_vector = []

        mode = self.training
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Retrieve intent and slot vocabularies from configuration.
            intent_labels = self.cfg.data_desc.intent_labels
            slot_labels = self.cfg.data_desc.slot_labels

            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            # Dataset.
            infer_datalayer = self._setup_infer_dataloader(queries, test_ds)

            for batch in infer_datalayer:
                input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask = batch

                intent_logits, slot_logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )

                # predict intents and slots for these examples
                # intents
                intent_preds = tensor2list(torch.sigmoid(intent_logits))
                # convert numerical outputs to Intent and Slot labels from the dictionaries
                for intents in intent_preds:
                    intent_lst = []
                    temp_list = []
                    for intent_num, probability in enumerate(intents):
                        if probability >= threshold:
                            intent_lst.append((intent_labels[int(intent_num)], round(probability, 2)))
                            temp_list.append(1)
                        else:
                            temp_list.append(0)

                    predicted_vector.append(temp_list)
                    predicted_intents.append(intent_lst)

                # slots
                slot_preds = torch.argmax(slot_logits, axis=-1)
                temp_slots_preds = []

                for slot_preds_query, mask_query in zip(slot_preds, subtokens_mask):
                    temp_slots = ""
                    query_slots = ""
                    for slot, mask in zip(slot_preds_query, mask_query):
                        if mask == 1:
                            if slot < len(slot_labels):
                                query_slots += slot_labels[int(slot)] + " "
                                temp_slots += f"{slot} "
                            else:
                                query_slots += "Unknown_slot "
                                temp_slots += "0 "
                    predicted_slots.append(query_slots.strip())
                    temp_slots_preds.append(temp_slots)

        finally:
            # set mode back to its original value
            self.train(mode=mode)

        return predicted_intents, predicted_slots, predicted_vector

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        To be added
        """
        result = []
        return result
