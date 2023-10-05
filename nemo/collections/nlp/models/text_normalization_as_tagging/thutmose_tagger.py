# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from time import perf_counter
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.text_normalization_as_tagging import (
    ThutmoseTaggerDataset,
    ThutmoseTaggerTestDataset,
    bert_example,
    tagging,
)
from nemo.collections.nlp.data.text_normalization_as_tagging.utils import read_label_map, read_semiotic_classes
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.token_classifier import TokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ["ThutmoseTaggerModel"]


@experimental
class ThutmoseTaggerModel(NLPModel):
    """
    BERT-based tagging model for ITN, inspired by LaserTagger approach.
    It maps spoken-domain input words to tags:
        KEEP, DELETE, or any of predefined replacement tags which correspond to a written-domain fragment.
    Example: one hundred thirty four -> _1 <DELETE> 3 4_
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "semiotic_logits": NeuralType(('B', 'T', 'D'), LogitsType()),
        }

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)

        label_map_file = self.register_artifact("label_map", cfg.label_map, verify_src_exists=True)
        semiotic_classes_file = self.register_artifact(
            "semiotic_classes", cfg.semiotic_classes, verify_src_exists=True
        )
        self.label_map = read_label_map(label_map_file)
        self.semiotic_classes = read_semiotic_classes(semiotic_classes_file)

        self.num_labels = len(self.label_map)
        self.num_semiotic_labels = len(self.semiotic_classes)
        self.id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in self.label_map.items()}
        self.id_2_semiotic = {semiotic_id: semiotic for semiotic, semiotic_id in self.semiotic_classes.items()}
        self.max_sequence_len = cfg.get('max_sequence_len', self.tokenizer.tokenizer.model_max_length)

        # setup to track metrics
        # we will have (len(self.semiotic_classes) + 1) labels
        # last one stands for WRONG (span in which the predicted tags don't match the labels)
        # this is needed to feed the sequence of classes to classification_report during validation
        label_ids = self.semiotic_classes.copy()
        label_ids["WRONG"] = len(self.semiotic_classes)
        self.tag_classification_report = ClassificationReport(
            len(self.semiotic_classes) + 1, label_ids=label_ids, mode='micro', dist_sync_on_step=True
        )
        self.tag_multiword_classification_report = ClassificationReport(
            len(self.semiotic_classes) + 1, label_ids=label_ids, mode='micro', dist_sync_on_step=True
        )
        self.semiotic_classification_report = ClassificationReport(
            len(self.semiotic_classes) + 1, label_ids=label_ids, mode='micro', dist_sync_on_step=True
        )

        self.hidden_size = cfg.hidden_size

        self.logits = TokenClassifier(
            self.hidden_size, num_classes=self.num_labels, num_layers=1, log_softmax=False, dropout=0.1
        )
        self.semiotic_logits = TokenClassifier(
            self.hidden_size, num_classes=self.num_semiotic_labels, num_layers=1, log_softmax=False, dropout=0.1
        )

        self.loss_fn = CrossEntropyLoss(logits_ndim=3)

        self.builder = bert_example.BertExampleBuilder(
            self.label_map, self.semiotic_classes, self.tokenizer.tokenizer, self.max_sequence_len
        )

    @typecheck()
    def forward(self, input_ids, input_mask, segment_ids):

        src_hiddens = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        tag_logits = self.logits(hidden_states=src_hiddens)
        semiotic_logits = self.semiotic_logits(hidden_states=src_hiddens)
        return tag_logits, semiotic_logits

    # Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """

        input_ids, input_mask, segment_ids, labels_mask, labels, semiotic_labels, _ = batch
        tag_logits, semiotic_logits = self.forward(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        loss_on_tags = self.loss_fn(logits=tag_logits, labels=labels, loss_mask=labels_mask)
        loss_on_semiotic = self.loss_fn(logits=semiotic_logits, labels=semiotic_labels, loss_mask=labels_mask)
        loss = loss_on_tags + loss_on_semiotic
        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)
        return {'loss': loss, 'lr': lr}

    # Validation and Testing
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_mask, segment_ids, labels_mask, tag_labels, semiotic_labels, semiotic_spans = batch
        tag_logits, semiotic_logits = self.forward(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        tag_preds = torch.argmax(tag_logits, dim=2)
        semiotic_preds = torch.argmax(semiotic_logits, dim=2)

        # Update tag classification_report
        predictions, labels = tag_preds.tolist(), tag_labels.tolist()
        for prediction, label, semiotic in zip(predictions, labels, semiotic_spans):
            # Here we want to track whether the predicted output matches ground truth labels
            # for each whole semiotic span
            # so we construct the special input for classification report, for example:
            #   label = [PLAIN, PLAIN, DATE, PLAIN, LETTERS, PLAIN]
            #   pred = [PLAIN, PLAIN, WRONG, PLAIN, LETTERS, PLAIN]
            span_labels = []
            span_predictions = []
            for cid, start, end in semiotic:
                if cid == -1:
                    break
                span_labels.append(cid)
                if prediction[start:end] == label[start:end]:
                    span_predictions.append(cid)
                else:
                    span_predictions.append(self.tag_classification_report.num_classes - 1)  # this stands for WRONG
            if len(span_labels) != len(span_predictions):
                raise ValueError(
                    "Length mismatch: len(span_labels)="
                    + str(len(span_labels))
                    + "; len(span_predictions)="
                    + str(len(span_predictions))
                )
            self.tag_classification_report(
                torch.tensor(span_predictions).to(self.device), torch.tensor(span_labels).to(self.device)
            )

            # We collect a separate classification_report for multiword replacements, as they are harder for the model
            multiword_span_labels = []
            multiword_span_predictions = []
            for cid, start, end in semiotic:
                if cid == -1:
                    break
                # this is a trick to determine if label consists of a single replacement
                # - it will be repeated for each input subtoken
                if len(set(label[start:end])) == 1:
                    continue
                multiword_span_labels.append(cid)
                if prediction[start:end] == label[start:end]:
                    multiword_span_predictions.append(cid)
                else:
                    # this stands for WRONG
                    multiword_span_predictions.append(self.tag_classification_report.num_classes - 1)
            if len(multiword_span_labels) != len(multiword_span_predictions):
                raise ValueError(
                    "Length mismatch: len(multiword_span_labels)="
                    + str(len(multiword_span_labels))
                    + "; len(multiword_span_predictions)="
                    + str(len(multiword_span_predictions))
                )

            self.tag_multiword_classification_report(
                torch.tensor(multiword_span_predictions).to(self.device),
                torch.tensor(multiword_span_labels).to(self.device),
            )

        # Update semiotic classification_report
        predictions, labels = semiotic_preds.tolist(), semiotic_labels.tolist()
        for prediction, label, semiotic in zip(predictions, labels, semiotic_spans):
            # Here we want to track whether the predicted output matches ground truth labels for whole semiotic span
            # so we construct the special input for classification report, for example:
            #   label = [PLAIN, PLAIN, DATE, PLAIN, LETTERS, PLAIN]
            #   pred = [PLAIN, PLAIN, WRONG, PLAIN, LETTERS, PLAIN]
            span_labels = []
            span_predictions = []
            for cid, start, end in semiotic:
                if cid == -1:
                    break
                span_labels.append(cid)
                if prediction[start:end] == label[start:end]:
                    span_predictions.append(cid)
                else:
                    span_predictions.append(self.tag_classification_report.num_classes - 1)  # this stands for WRONG
            if len(span_labels) != len(span_predictions):
                raise ValueError(
                    "Length mismatch: len(span_labels)="
                    + str(len(span_labels))
                    + "; len(span_predictions)="
                    + str(len(span_predictions))
                )
            self.semiotic_classification_report(
                torch.tensor(span_predictions).to(self.device), torch.tensor(span_labels).to(self.device)
            )

        val_loss_tag = self.loss_fn(logits=tag_logits, labels=tag_labels, loss_mask=labels_mask)
        val_loss_semiotic = self.loss_fn(logits=semiotic_logits, labels=semiotic_labels, loss_mask=labels_mask)
        val_loss = val_loss_tag + val_loss_semiotic
        self.validation_step_outputs.append({'val_loss': val_loss})
        return {'val_loss': val_loss}

    def on_validation_epoch_end(self):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()

        # calculate metrics and classification report
        # In our task recall = accuracy, and the recall column - is the per class accuracy
        _, tag_accuracy, _, tag_report = self.tag_classification_report.compute()
        _, tag_multiword_accuracy, _, tag_multiword_report = self.tag_multiword_classification_report.compute()

        _, semiotic_accuracy, _, semiotic_report = self.semiotic_classification_report.compute()

        logging.info("Total tag accuracy: " + str(tag_accuracy))
        logging.info(tag_report)
        logging.info("Only multiword tag accuracy: " + str(tag_multiword_accuracy))
        logging.info(tag_multiword_report)

        logging.info("Total semiotic accuracy: " + str(semiotic_accuracy))
        logging.info(semiotic_report)

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('tag accuracy', tag_accuracy)
        self.log('tag multiword accuracy', tag_multiword_accuracy)
        self.log('semiotic accuracy', semiotic_accuracy)

        self.tag_classification_report.reset()
        self.tag_multiword_classification_report.reset()
        self.semiotic_classification_report.reset()

        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.on_validation_epoch_end()

    # Functions for inference
    @torch.no_grad()
    def _infer(self, sents: List[str]) -> List[List[int]]:
        """ Main function for Inference

        Args:
            sents: A list of input sentences (lowercase spoken-domain words separated by space).

        Returns:
            all_preds: A list of tab-separated text records, same size as input list. Each record consists of 4 items:
                - final output text
                - input words
                - tags predicted for input words
                - tags after swap preprocessing
                - semiotic labels predicted for input words
        """

        # all input sentences go into one batch
        dataloader_cfg = {"batch_size": len(sents), "num_workers": 3, "pin_memory": False}
        infer_datalayer = self._setup_infer_dataloader(dataloader_cfg, sents)

        batch = next(iter(infer_datalayer))
        input_ids, input_mask, segment_ids = batch

        tag_logits, semiotic_logits = self.forward(
            input_ids=input_ids.to(self.device),
            input_mask=input_mask.to(self.device),
            segment_ids=segment_ids.to(self.device),
        )

        all_preds = []
        for i, sent in enumerate(sents):
            example = self.builder.build_bert_example(source=sent, infer=True)
            tag_preds = tensor2list(torch.argmax(tag_logits[i], dim=-1))
            semiotic_preds = tensor2list(torch.argmax(semiotic_logits[i], dim=-1))

            # this mask is required by get_token_labels
            example.features["labels_mask"] = [0] + [1] * (len(semiotic_preds) - 2) + [0]
            example.features["tag_labels"] = tag_preds
            example.features["semiotic_labels"] = semiotic_preds
            tags = [self.id_2_tag[label_id] for label_id in example.get_token_labels("tag_labels")]
            semiotic_labels = [
                self.id_2_semiotic[label_id] for label_id in example.get_token_labels("semiotic_labels")
            ]

            prediction, inp_str, tag_str, tag_with_swap_str = example.editing_task.realize_output(
                tags, semiotic_labels
            )
            all_preds.append(
                prediction
                + "\t"
                + inp_str
                + "\t"
                + tag_str
                + "\t"
                + tag_with_swap_str
                + "\t"
                + " ".join(semiotic_labels)
            )

        return all_preds

    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, data_split="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, data_split="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.data_path is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, data_split="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, data_split: str):
        start_time = perf_counter()
        logging.info(f'Creating {data_split} dataset')
        input_file = cfg.data_path
        dataset = ThutmoseTaggerDataset(input_file=input_file, example_builder=self.builder)
        dl = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, collate_fn=dataset.collate_fn
        )
        running_time = perf_counter() - start_time
        logging.info(f'Took {running_time} seconds')
        return dl

    def _setup_infer_dataloader(self, cfg: DictConfig, queries: List[str]) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Args:
            cfg: config dictionary containing data loader params like batch_size, num_workers and pin_memory
            queries: text
        Returns:
            A pytorch DataLoader.
        """
        dataset = ThutmoseTaggerTestDataset(sents=queries, example_builder=self.builder)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        result = [
            PretrainedModelInfo(
                pretrained_model_name="itn_en_thutmose_bert",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/itn_en_thutmose_bert/versions/1.9.0/files/itn_en_thutmose_bert.nemo",
                description="A single-pass tagger-based English model for inverse text normalization based"
                "on BERT, trained on 2 mln sentences from Google Text Normalization Dataset",
            ),
            PretrainedModelInfo(
                pretrained_model_name="itn_ru_thutmose_bert",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/itn_ru_thutmose_bert/versions/1.11.0/files/itn_ru_thutmose_bert.nemo",
                description="A single-pass tagger-based Russian model for inverse text normalization based"
                "on BERT, trained on 2 mln sentences from Google Text Normalization Dataset",
            ),
        ]
        return result
