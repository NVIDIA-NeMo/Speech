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


import os
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.text_classification import TextClassificationDataset, calc_class_weights
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import SequenceClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.utils import logging

__all__ = ['TextClassificationModel']


class TextClassificationModel(NLPModel, Exportable):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the BERTTextClassifier model."""
        # shared params for dataset and data loaders
        self.dataset_cfg = cfg.dataset
        self.class_weights = None

        super().__init__(cfg=cfg, trainer=trainer)

        self.classifier = SequenceClassifier(
            hidden_size=self.hidden_size,
            num_classes=cfg.dataset.num_classes,
            num_layers=cfg.classifier_head.num_output_layers,
            activation='relu',
            log_softmax=False,
            dropout=cfg.classifier_head.fc_dropout,
            use_transformer_init=True,
            idx_conditioned_on=0,
        )

        self.create_loss_module()

        # setup to track metrics
        self.classification_report = ClassificationReport(
            num_classes=cfg.dataset.num_classes, mode='micro', dist_sync_on_step=True
        )

        # register the file containing the labels into the artifacts to get stored in the '.nemo' file later
        if 'class_labels' in cfg and 'class_labels_file' in cfg.class_labels and cfg.class_labels.class_labels_file:
            self.register_artifact('class_labels.class_labels_file', cfg.class_labels.class_labels_file)

    def create_loss_module(self):
        # create the loss module if it is not yet created by the training data loader
        if not hasattr(self, 'loss'):
            if hasattr(self, 'class_weights') and self.class_weights:
                # You may need to increase the number of epochs for convergence when using weighted_loss
                self.loss = CrossEntropyLoss(weight=self.class_weights)
            else:
                self.loss = CrossEntropyLoss()

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        logits = self.classifier(hidden_states=hidden_states)
        return logits.float()

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        train_loss = self.loss(logits=logits, labels=labels)

        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': train_loss,
            'lr': lr,
        }

    def validation_step(self, batch, batch_idx, split="val"):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=logits, labels=labels)

        preds = torch.argmax(logits, axis=-1)

        tp, fn, fp, _ = self.classification_report(preds, labels)

        loss = {f'{split}_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}
        if split == 'val':
            self.validation_step_outputs.append(loss)
        elif split == 'test':
            self.test_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self, split="val"):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.tensor(0)
        if split == 'val':
            avg_loss = torch.stack([x[f'val_loss'] for x in self.validation_step_outputs]).mean()
            self.validation_step_outputs.clear()  # free memory
        elif split == 'test':
            avg_loss = torch.stack([x[f'test_loss'] for x in self.test_step_outputs]).mean()
            self.test_step_outputs.clear()  # free memory

        # calculate metrics and classification report
        precision, recall, f1, report = self.classification_report.compute()

        logging.info(f'{split}_report: {report}')

        self.log(f'{split}_loss', avg_loss, prog_bar=True)
        self.log(f'{split}_precision', precision)
        self.log(f'{split}_f1', f1)
        self.log(f'{split}_recall', recall)

        self.classification_report.reset()

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, 'test')

    def on_test_epoch_end(self):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.on_validation_epoch_end(split='test')

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

        # calculate the class weights to be used in the loss function
        if self.cfg.dataset.class_balancing == 'weighted_loss':
            self.class_weights = calc_class_weights(train_data_config.file_path, self.cfg.dataset.num_classes)
        else:
            self.class_weights = None
        # we need to create/update the loss module by using the weights calculated from the training data
        self.create_loss_module()

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or not test_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: Dict) -> 'torch.utils.data.DataLoader':
        input_file = cfg.file_path
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f'{input_file} not found! The data should be be stored in TAB-separated files \n\
                "validation_ds.file_path" and "train_ds.file_path" for train and evaluation respectively. \n\
                Each line of the files contains text sequences, where words are separated with spaces. \n\
                The label of the example is separated with TAB at the end of each line. \n\
                Each line of the files should follow the format: \n\
                [WORD][SPACE][WORD][SPACE][WORD][...][TAB][LABEL]'
            )

        dataset = TextClassificationDataset(
            tokenizer=self.tokenizer,
            input_file=input_file,
            max_seq_length=self.dataset_cfg.max_seq_length,
            num_samples=cfg.get("num_samples", -1),
            shuffle=cfg.shuffle,
            use_cache=self.dataset_cfg.use_cache,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            collate_fn=dataset.collate_fn,
        )

    @torch.no_grad()
    def classifytext(self, queries: List[str], batch_size: int = 1, max_seq_length: int = -1) -> List[int]:
        """
        Get prediction for the queries
        Args:
            queries: text sequences
            batch_size: batch size to use during inference
            max_seq_length: sequences longer than max_seq_length will get truncated. default -1 disables truncation.
        Returns:
            all_preds: model predictions
        """
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        device = next(self.parameters()).device
        try:
            # Switch model to evaluation mode
            self.eval()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            dataloader_cfg = {"batch_size": batch_size, "num_workers": 3, "pin_memory": False}
            infer_datalayer = self._setup_infer_dataloader(dataloader_cfg, queries, max_seq_length)

            for i, batch in enumerate(infer_datalayer):
                input_ids, input_type_ids, input_mask, subtokens_mask = batch

                logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )

                preds = tensor2list(torch.argmax(logits, axis=-1))
                all_preds.extend(preds)
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            logging.set_verbosity(logging_level)
        return all_preds

    def _setup_infer_dataloader(
        self, cfg: Dict, queries: List[str], max_seq_length: int = -1
    ) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.

        Args:
            cfg: config dictionary containing data loader params like batch_size, num_workers and pin_memory
            queries: text
            max_seq_length: maximum length of queries, default is -1 for no limit
        Returns:
            A pytorch DataLoader.
        """
        dataset = TextClassificationDataset(tokenizer=self.tokenizer, queries=queries, max_seq_length=max_seq_length)
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
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
