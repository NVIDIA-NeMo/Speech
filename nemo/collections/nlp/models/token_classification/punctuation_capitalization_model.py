# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import copy
import warnings
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from tqdm import tqdm

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    BertPunctuationCapitalizationDataset,
    PunctuationCapitalizationEvalDataConfig,
    PunctuationCapitalizationTrainDataConfig,
    load_label_ids,
    raise_not_equal_labels_error,
)
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_infer_dataset import (
    BertPunctuationCapitalizationInferDataset,
)
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset import (
    BertPunctuationCapitalizationTarredDataset,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.models.token_classification.punctuation_capitalization_config import (
    is_legacy_model_config,
    legacy_model_config_to_new_model_config,
)
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils import logging

__all__ = ['PunctuationCapitalizationModel']


class PunctuationCapitalizationModel(NLPModel, Exportable):
    """
    A model for restoring punctuation and capitalization in text. The model is usually used together with ASR model
    because ASR models often return text without punctuation and capitalization.

    The model consists of a language model and two multilayer perceptrons (MLP) on top the language model. The first
    MLP serves for punctuation prediction and the second is for capitalization prediction. You can use only BERT-like
    HuggingFace language models (model ``forward`` method accepts ``input_ids``, ``token_types_ids``,
    ``attention_mask`` arguments). See more about model config options :ref:`here<model-config-label>`.

    Use method :meth:`~add_punctuation_capitalization` for model inference.

    For training and testing use dataset
    :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset`,
    for training on huge amounts of data which cannot be loaded into memory simultaneously use
    :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.BertPunctuationCapitalizationTarredDataset`.

    Args:
        cfg: a model configuration. It should follow dataclass
            :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_config.PunctuationCapitalizationModelConfig`
            See an example of full config in
            `nemo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml
            <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`_
        trainer: an instance of a PyTorch Lightning trainer
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Neural types of a :meth:`forward` method output."""
        return {
            "punct_logits": NeuralType(('B', 'T', 'C'), LogitsType()),
            "capit_logits": NeuralType(('B', 'T', 'C'), LogitsType()),
        }

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        """Initializes BERT Punctuation and Capitalization model."""
        if is_legacy_model_config(cfg):
            cfg = legacy_model_config_to_new_model_config(cfg)

        # For structure of `self.metrics` attribute see `self._setup_metrics_dictionary` method.
        self.metrics: Optional[torch.nn.ModuleDict] = None
        self.label_ids_are_set: bool = False
        self.punct_label_ids: Optional[Dict[str, int]] = None
        self.capit_label_ids: Optional[Dict[str, int]] = None
        super().__init__(cfg=cfg, trainer=trainer)
        if not self.label_ids_are_set:
            self._set_label_ids()

        self.punct_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=len(self.punct_label_ids),
            activation=cfg.punct_head.activation,
            log_softmax=False,
            dropout=cfg.punct_head.fc_dropout,
            num_layers=cfg.punct_head.num_fc_layers,
            use_transformer_init=cfg.punct_head.use_transformer_init,
        )

        self.capit_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=len(self.capit_label_ids),
            activation=cfg.capit_head.activation,
            log_softmax=False,
            dropout=cfg.capit_head.fc_dropout,
            num_layers=cfg.capit_head.num_fc_layers,
            use_transformer_init=cfg.capit_head.use_transformer_init,
        )

        self.loss = CrossEntropyLoss(logits_ndim=3)
        self.agg_loss = AggregatorLoss(num_inputs=2)

    @typecheck()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a forward pass through the model. For more details see ``forward`` method of HuggingFace BERT-like
        (models which accept ``input_ids``, ``attention_mask``, ``token_type_ids`` arguments) models.

        Args:
            input_ids (:obj:`torch.Tensor`): an integer torch tensor of shape ``[Batch, Time]``. Contains encoded
                source tokens.
            attention_mask (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. Contains an
                attention mask for excluding paddings.
            token_type_ids (:obj:`torch.Tensor`): an integer torch Tensor of shape ``[Batch, Time]``. Contains an index
                of segment to which a token belongs. If ``token_type_ids`` is not ``None``, then it should be a zeros
                tensor.

        Returns:
            :obj:`Tuple[torch.Tensor, torch.Tensor]`: a tuple containing

                - ``punct_logits`` (:obj:`torch.Tensor`): a float torch tensor of shape
                  ``[Batch, Time, NumPunctuationLabels]`` containing punctuation logits
                - ``capit_logits`` (:obj:`torch.Tensor`): a float torch tensor of shape
                  ``[Batch, Time, NumCapitalizationLabels]`` containing capitalization logits
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        punct_logits = self.punct_classifier(hidden_states=hidden_states)
        capit_logits = self.capit_classifier(hidden_states=hidden_states)
        return punct_logits.float(), capit_logits.float()

    def _make_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        punct_logits, capit_logits = self(
            input_ids=batch['input_ids'], token_type_ids=batch['segment_ids'], attention_mask=batch['input_mask']
        )

        punct_loss = self.loss(logits=punct_logits, labels=batch['punct_labels'], loss_mask=batch['loss_mask'])
        capit_loss = self.loss(logits=capit_logits, labels=batch['capit_labels'], loss_mask=batch['loss_mask'])
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        return loss, punct_logits, capit_logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Lightning calls this inside the training loop with the data from the training dataloader passed in as
        ``batch``.

        Args:
            batch: a dictionary with following
                items:

                  - ``'input_ids'`` (:obj:`torch.Tensor`): an integer torch tensor of shape ``[Batch, Time]`` containing
                    encoded source text
                  - ``'segment_ids'`` (:obj:`torch.Tensor`): a zeros integer torch tensor of shape ``[Batch, Time]``
                  - ``'input_mask'`` (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. Serves as
                    attention mask. should be ``False`` on padding tokens and ``True`` on other tokens.
                  - ``'loss_mask'`` (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. Which token
                    to compute loss on. See more details in description of parameters ``ignore_start_end`` and
                    ``ignore_extra_tokens`` of a class
                    :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset`
                  - ``'punct_labels'`` (:obj:`torch.Tensor`): a ``long`` torch tensor of shape ``[Batch, Time]``.
                    Contains encoded punctuation labels
                  - ``'capit_labels'`` (:obj:`torch.Tensor`): a ``long`` torch tensor of shape ``[Batch, Time]``.
                    Contains encoded capitalization labels
                  - ``'subtokens_mask'`` (:obj:`torch.Tensor`): not required for training and can be omitted

            batch_idx (:obj:`int`): an index of batch. Mandatory Lightning parameter

        Returns:
            :obj:`Dict[str, Union[torch.Tensor, float]]`: a dictionary with 2 items:

                - ``'loss'`` (:obj:`torch.Tensor`): torch tensor containing mean aggregated punctuation and
                  capitalization loss
                - ``'lr'`` (:obj:`float`): a float containing learning rate
        """
        loss, _, _ = self._make_step(batch)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)
        self.log('train_loss', loss)
        return {'loss': loss, 'lr': lr}

    def eval_step(self, batch: Dict[str, torch.Tensor], mode: str, dataloader_idx: int) -> Dict[str, None]:
        """
        A method called by :meth:`validation_step` and :meth:`test_step`. Performs forward pass and updates metrics.

        Args:
            batch (:obj:`Dict[str, torch.Tensor]`): a dictionary with following items:

                - ``'input_ids'`` (:obj:`torch.Tensor`): an integer torch tensor of shape ``[Batch, Time]`` containing
                  encoded source text.
                - ``'subtokens_mask'`` (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. An
                  element of this item is ``True`` if corresponding token from ``'input_ids'`` element is the first
                  token in some word.
                - ``'segment_ids'`` (:obj:`torch.Tensor`): a zeros integer torch tensor of shape ``[Batch, Time]``.
                - ``'input_mask'`` (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. Serves as
                  attention mask. should be ``False`` on padding tokens and ``True`` on other tokens.
                - ``'loss_mask'`` (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. Which token
                  to compute loss on. See more details in description of parameters ``ignore_start_end`` and
                  ``ignore_extra_tokens`` of class
                  :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset`.
                - ``'punct_labels'`` (:obj:`torch.Tensor`): a long torch tensor of shape ``[Batch, Time]``. Contains
                  encoded punctuation labels.
                - ``'capit_labels'`` (:obj:`torch.Tensor`): a long torch tensor of shape ``[Batch, Time]``. Contains
                  encoded capitalization labels.
            mode: either ``'validation'`` or ``'test'`` depending on caller method.
            dataloader_idx: NeMo parameter for multi dataset validation.

        Returns:
            :obj:`Dict[str, None]`: a dictionary containing items ``'loss'``, ``'punct_class_report'``,
            ``'capit_class_report'`` which values are ``None``. Values are ``None`` because metrics are computed using
            ``torchmetrics``.
        """
        loss, punct_logits, capit_logits = self._make_step(batch)
        subtokens_mask = batch['subtokens_mask']
        punct_preds = torch.argmax(punct_logits, axis=-1)[subtokens_mask]
        punct_labels = batch['punct_labels'][subtokens_mask]
        capit_preds = torch.argmax(capit_logits, axis=-1)[subtokens_mask]
        capit_labels = batch['capit_labels'][subtokens_mask]
        self.metrics[mode]['loss'][dataloader_idx](
            loss=loss, num_measurements=batch['loss_mask'].sum().to(loss.device)
        )
        self.metrics[mode]['punct_class_report'][dataloader_idx](punct_preds, punct_labels)
        self.metrics[mode]['capit_class_report'][dataloader_idx](capit_preds, capit_labels)
        # torchmetrics are used for metrics computation
        return {'loss': None, 'punct_class_report': None, 'capit_class_report': None}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, None]:
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader passed in as
        ``batch``. See more details in :meth:`eval_step`.

        Args:
            batch (:obj:`dict`): see :meth:`eval_step` for the ``batch`` parameter explanation
            batch_idx (:obj:`int`): an index of a batch in a dataset. A mandatory Lightning parameter
            dataloader_idx (:obj:`int`): a NeMo parameter for performing testing on multiple datasets

        Returns:
            :obj:`Dict[str, None]`: a dictionary containing items ``'loss'``, ``'punct_class_report'``,
            ``'capit_class_report'`` which values are ``None``. Values are ``None`` because metrics are computed using
            ``torchmetrics``.
        """
        loss = self.eval_step(batch, 'val', dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(loss)
        else:
            self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, None]:
        """
        Lightning calls this inside the test loop with the data from the test dataloader passed in as ``batch``.
        See more details in :meth:`eval_step`.

        Args:
            batch (:obj:`dict`): see :meth:`eval_step` for the ``batch`` parameter explanation
            batch_idx (:obj:`int`): an index of a batch in a dataset. A mandatory Lightning parameter
            dataloader_idx (:obj:`int`): a NeMo parameter for performing testing on multiple datasets

        Returns:
            :obj:`Dict[str, None]`: a dictionary containing items ``'loss'``, ``'punct_class_report'``,
            ``'capit_class_report'`` which values are ``None``. Values are ``None`` because metrics are computed using
            ``torchmetrics``.
        """
        loss = self.eval_step(batch, 'test', dataloader_idx)
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(loss)
        else:
            self.test_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Called at the end of training epoch. This method properly shuffles
        :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset`.
        Regular data loader shuffling only permutes batches.

        Args:
            outputs (:obj:`pytorch_lightning.utilities.types.EPOCH_OUTPUT`): an output of all training steps. It is a
                mandatory PyTorch Lightning parameter, and it is not used in this method
        """
        shuffle = self._cfg.train_ds.get('shuffle')
        if shuffle is None:  # Encountered legacy config
            shuffle = not self.cfg.train_ds.get('use_tarred_dataset', False)
        if shuffle:
            if isinstance(self.train_dataloader().dataset, BertPunctuationCapitalizationDataset):
                self.train_dataloader().dataset.repack_batches_with_shuffle()

    def _multi_eval_epoch_end(self, mode: str, dataloader_idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        loss = self.metrics[mode]['loss'][dataloader_idx].compute()
        self.metrics[mode]['loss'][dataloader_idx].reset()

        punct_res = self.metrics[mode]['punct_class_report'][dataloader_idx].compute()
        punct_precision, punct_recall, punct_f1, punct_report = punct_res
        self.metrics[mode]['punct_class_report'][dataloader_idx].reset()

        capit_res = self.metrics[mode]['capit_class_report'][dataloader_idx].compute()
        capit_precision, capit_recall, capit_f1, capit_report = capit_res
        self.metrics[mode]['capit_class_report'][dataloader_idx].reset()
        log_dict = {
            'log': {
                f'{mode}_loss': loss,
                f'{mode}_punct_precision': punct_precision,
                f'{mode}_punct_f1': punct_f1,
                f'{mode}_punct_recall': punct_recall,
                f'{mode}_capit_precision': capit_precision,
                f'{mode}_capit_f1': capit_f1,
                f'{mode}_capit_recall': capit_recall,
            }
        }
        logging.info(f'Punctuation report: {punct_report}')
        logging.info(f'Capitalization report: {capit_report}')
        return log_dict

    def multi_validation_epoch_end(self, outputs: Any, dataloader_idx: int = 0) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Called at the end of validation to compute and log metrics.
        """
        return self._multi_eval_epoch_end('val', dataloader_idx)

    def multi_test_epoch_end(self, outputs: Any, dataloader_idx: int = 0) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Called at the end of model testing to compute and log metrics.
        """
        return self._multi_eval_epoch_end('test', dataloader_idx)

    def update_config_after_restoring_from_checkpoint(self, **kwargs) -> None:
        """
        Set new values for some sections of config. Useful after restoring from checkpoint for fine-tuning
        and testing if config parameters of a restored checkpoint are not suitable.

        For ``class_labels``, ``common_dataset_parameters``, ``train_ds``, ``validation_ds``, ``test_ds``, there is
        no need to provide values for all items in an updated config section. If an item is omitted in this method
        parameter, then corresponding item in model config does not change.

        If the entire updated section is missing in the model config, then omitted items from this method parameters
        are set according to default values listed
        :ref:`here <run-config-label>`.

        .. warning::
            Parameter ``optim`` is processed in a special way. ``optim`` contents are used not for updating of
            model config, but for replacement of entire config section.

        If one of parameters ``train_ds``, ``validation_ds``, ``test_ds``, is provided but its value is
        ``None``, then corresponding section is replaced with ``None``.

        .. warning::
            You may change values of parameters related to label ids:

                - ``common_dataset_parameters.punct_label_ids``,
                - ``common_dataset_parameters.capit_label_ids``,
                - ``common_dataset_parameters.label_vocab_dir``,
                - ``class_labels.punct_labels_file``,
                - ``class_labels.capit_labels_file``,

            yet label ids in these parameters must be equal to label ids loaded from checkpoint. Otherwise,
            an error will be raised.

        Keyword Args:
            class_labels (:obj:`Union[DictConfig, Dict[str, str]]`): names of label id files used as label
                id dictionaries. See more in :ref:`class labels' config<class-labels-config-label>`.
            common_dataset_parameters (:obj:`Union[DictConfig, Dict[str, Any]]`, `optional`): see more in
                :ref:`common dataset parameters config<common-dataset-parameters-config-label>`.
            train_ds (:obj:`Union[DictConfig, Dict[str, Any]]`, `optional`): configuration of training dataset. See
                possible options in :ref:`data config<data-config-label>`.
            validation_ds (:obj:`Union[DictConfig, Dict[str, Any]]`, `optional`): configuration of validation
                dataset. See possible options in :ref:`data config<data-config-label>`.
            test_ds (:obj:`Union[DictConfig, Dict[str, Any]]`, `optional`): configuration of test dataset. See
                possible options in :ref:`data config<data-config-label>`.
            optim (:obj:`Union[DictConfig, Dict[str, Any]]`, `optional`): optimization configuration. See possible
                options in :ref:`optimization<optimization-label>` and in `primer
                <https://github.com/NVIDIA/NeMo/blob/main/tutorials/00_NeMo_Primer.ipynb>`_ tutorial.
        """
        allowed_keys = {'class_labels', 'common_dataset_parameters', 'train_ds', 'validation_ds', 'test_ds', 'optim'}
        unexpected_keys = set(kwargs) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                f"Found unexpected keyword arguments: {unexpected_keys}. You can use only {allowed_keys}."
            )
        if 'class_labels' in kwargs:
            if kwargs['class_labels'] is None:
                raise ValueError(
                    f"'class_labels' parameters is `None`, whereas you cannot remove section 'class_labels' from model "
                    f"config."
                )
            self._cfg.class_labels = OmegaConf.merge(self._cfg.class_labels, OmegaConf.create(kwargs['class_labels']))
        if 'common_dataset_parameters' in kwargs:
            if kwargs['common_dataset_parameters'] is None:
                raise ValueError(
                    f"'common_dataset_parameters' item is `None`, whereas you cannot remove section"
                    f"'common_dataset_parameters' from model config."
                )
            self._cfg.common_dataset_parameters = OmegaConf.merge(
                self._cfg.common_dataset_parameters, OmegaConf.create(kwargs['common_dataset_parameters'])
            )
            self._check_label_config_parameters()
        if 'train_ds' in kwargs:
            if kwargs['train_ds'] is None:
                self._cfg.train_ds = None
            else:
                if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                    base = self._cfg.train_ds
                else:
                    base = OmegaConf.structured(PunctuationCapitalizationTrainDataConfig)
                self._cfg.train_ds = OmegaConf.merge(base, OmegaConf.create(kwargs['train_ds']))
        if 'validation_ds' in kwargs:
            if kwargs['validation_ds'] is None:
                self._cfg.validation_ds = None
            else:
                if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                    base = self._cfg.validation_ds
                else:
                    base = OmegaConf.structured(PunctuationCapitalizationEvalDataConfig)
                self._cfg.validation_ds = OmegaConf.merge(base, OmegaConf.create(kwargs['validation_ds']))
        if 'test_ds' in kwargs:
            if kwargs['test_ds'] is None:
                self._cfg.test_ds = None
            else:
                if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                    base = self._cfg.test_ds
                else:
                    base = OmegaConf.structured(PunctuationCapitalizationEvalDataConfig)
                self._cfg.test_ds = OmegaConf.merge(base, OmegaConf.create(kwargs['test_ds']))
        if 'optim' in kwargs:
            self._cfg.optim = kwargs['optim']

    def setup_training_data(self, train_data_config: Optional[Union[Dict[str, Any], DictConfig]] = None) -> None:
        """
        Sets up training data: creates dataset and sets data loader. If parameter ``train_data_config`` is not
        provided, then :ref:`config<model-config-label>` section ``train_ds`` will be used.

        Args:
            train_data_config (:obj:`Union[Dict[str, Any], DictConfig]`, `optional`): a dictionary that should contain
                only fields present in :ref:`data config<data-config-label>`.
                If some of the fields are missing, then they will be set according to
                :ref:`data config<data-config-label>` defaults. If ``train_data_config`` parameter is not set, then
                ``train_ds`` item of model config is used. Here model config is a configuration used for model
                instantiation.
        """
        if train_data_config is not None:
            train_data_config = OmegaConf.create(train_data_config)
            train_data_config = OmegaConf.merge(
                OmegaConf.structured(PunctuationCapitalizationTrainDataConfig), train_data_config
            )
        if train_data_config is None:
            train_data_config = self._cfg.train_ds

        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, train=True)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'use_tarred_dataset' in train_data_config and train_data_config['use_tarred_dataset']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches * ceil(len(self._train_dl.dataset) / self.world_size)
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

        self.punct_label_ids = self._train_dl.dataset.punct_label_ids.copy()
        self.capit_label_ids = self._train_dl.dataset.capit_label_ids.copy()
        self.label_ids_are_set = True
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            label_vocab_dir = self._cfg.common_dataset_parameters.label_vocab_dir
            if label_vocab_dir is None:
                punct_label_ids_file, capit_label_ids_file = self._train_dl.dataset.save_labels_and_get_file_paths(
                    self._cfg.class_labels.punct_labels_file, self._cfg.class_labels.capit_labels_file
                )
            else:
                punct_label_ids_file = Path(label_vocab_dir).expanduser() / self._cfg.class_labels.punct_labels_file
                capit_label_ids_file = Path(label_vocab_dir).expanduser() / self._cfg.class_labels.capit_labels_file
            self.register_artifact('class_labels.punct_labels_file', str(punct_label_ids_file))
            self.register_artifact('class_labels.capit_labels_file', str(capit_label_ids_file))

    def _get_eval_metrics_kwargs(
        self,
    ) -> Tuple[
        Dict[str, bool],
        Dict[str, Union[bool, str, int, Dict[str, int]]],
        Dict[str, Union[bool, str, int, Dict[str, int]]],
    ]:
        loss_kw = {'dist_sync_on_step': False, 'take_avg_loss': True}
        punct_kw = {
            'num_classes': len(self.punct_label_ids),
            'label_ids': self.punct_label_ids,
            'mode': 'macro',
            'dist_sync_on_step': False,
        }
        capit_kw = {
            'num_classes': len(self.capit_label_ids),
            'label_ids': self.capit_label_ids,
            'mode': 'macro',
            'dist_sync_on_step': False,
        }
        return loss_kw, punct_kw, capit_kw

    def _setup_metrics_dictionary(self) -> None:
        eval_metrics = torch.nn.ModuleDict(
            {
                "loss": torch.nn.ModuleList([]),
                "punct_class_report": torch.nn.ModuleList([]),
                "capit_class_report": torch.nn.ModuleList([]),
            }
        )
        self.metrics = torch.nn.ModuleDict({"val": eval_metrics, "test": copy.deepcopy(eval_metrics)})

    def setup_validation_data(self, val_data_config: Optional[Union[Dict[str, Any], DictConfig]] = None) -> None:
        """
        Sets up validation data: creates dataset and sets data loader. If parameter ``val_data_config`` is not
        provided, then ``validation_ds`` :ref:`config <model-config-label>` section will be used. Here model config is
        a configuration used for model instantiation.

        Args:
            val_data_config (:obj:`Union[Dict[str, Any], DictConfig]`, `optional`): a dictionary that should contain
                only fields present in data config :ref:`description<data-config-label>`.
                If some of the fields are missing, then they will be set according to data config
                :ref:`description<data-config-label>` defaults. If ``val_data_config`` parameter is not set, then
                ``validation_ds`` item of model config is used. Here model config is a configuration used for model
                instantiation.
        """
        if val_data_config is not None:
            val_data_config = OmegaConf.create(val_data_config)
            val_data_config = OmegaConf.merge(
                OmegaConf.structured(PunctuationCapitalizationEvalDataConfig), val_data_config
            )
        if self.metrics is None:
            self._setup_metrics_dictionary()
        if val_data_config is None:
            val_data_config = self._cfg.validation_ds

        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, train=False)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'use_tarred_dataset' in val_data_config and val_data_config['use_tarred_dataset']:
            # We also need to check if limit_val_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # validation batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_val_batches, float):
                self._trainer.limit_val_batches = int(
                    self._trainer.limit_val_batches * ceil(len(self._validation_dl.dataset) / self.world_size)
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "validation batches will be used. Please set the trainer and rebuild the dataset."
                )

        loss_kw, punct_kw, capit_kw = self._get_eval_metrics_kwargs()
        self.metrics['val']['loss'].append(GlobalAverageLossMetric(**loss_kw))
        self.metrics['val']['punct_class_report'].append(ClassificationReport(**punct_kw))
        self.metrics['val']['capit_class_report'].append(ClassificationReport(**capit_kw))

    def setup_test_data(self, test_data_config: Optional[Union[Dict[str, Any], DictConfig]] = None) -> None:
        """
        Sets up test data: creates dataset and sets data loader. If parameter ``test_data_config`` is not
        provided, then ``test_ds`` config section will be used. See more about in data config
        :ref:`description <data-config-label>` and model config :ref:`description<model-config-label>`.

        Args:
            test_data_config (:obj:`Union[Dict[str, Any], DictConfig]`, `optional`): a dictionary that should contain
                only fields present in data config :ref:`description<data-config-label>`.
                If some of the fields are missing, then they will be set according to data config
                :ref:`description <data-config-label>` defaults. If ``test_data_config`` parameter is not set, then
                ``test_ds`` item of :ref:`model config <model-config-label>` is used. Here model config is a
                configuration used for model instantiation.
        """
        if test_data_config is not None:
            test_data_config = OmegaConf.create(test_data_config)
            test_data_config = OmegaConf.merge(
                OmegaConf.structured(PunctuationCapitalizationEvalDataConfig), test_data_config
            )
        if self.metrics is None:
            self._setup_metrics_dictionary()
        if test_data_config is None:
            test_data_config = self._cfg.test_ds
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, train=False)
        # Check for multiple dataloaders here as it may not get called in ModelPT when models are being restored
        if type(self._test_dl) == list and len(self._test_dl) > 1:
            for _ in range(len(self._test_dl)):
                self.test_step_outputs.append([])

        loss_kw, punct_kw, capit_kw = self._get_eval_metrics_kwargs()
        self.metrics['test']['loss'].append(GlobalAverageLossMetric(**loss_kw))
        self.metrics['test']['punct_class_report'].append(ClassificationReport(**punct_kw))
        self.metrics['test']['capit_class_report'].append(ClassificationReport(**capit_kw))

    def _check_label_config_parameters(self) -> None:
        """
        Checks that config items ``common_dataset_parameters.punct_label_ids`` and
        ``common_dataset_parameters.punct_label_vocab_file``,
        ``common_dataset_parameters.capit_label_ids`` and ``common_dataset_parameters.capit_label_vocab_file`` contain
        identical label ids. Of course, if any of these parameters is ``None``, then check is not performed.

        In addition, this method checks that ``common_dataset_parameters.pad_label`` has id ``0`` in punctuation and
        capitalization label ids.
        """
        pli = self._cfg.common_dataset_parameters.punct_label_ids
        cli = self._cfg.common_dataset_parameters.capit_label_ids
        pad_label = self._cfg.common_dataset_parameters.pad_label
        plvf, clvf = self._extract_label_vocab_files_from_config()
        for label_ids, label_vocab_file, already_set_label_ids, label_ids_name, label_vocab_name in [
            (pli, plvf, self.punct_label_ids, 'punct_label_ids', 'punct_label_vocab_file'),
            (cli, clvf, self.capit_label_ids, 'capit_label_ids', 'capit_label_vocab_file'),
        ]:
            if label_vocab_file is not None:
                file_label_ids = load_label_ids(label_vocab_file)
            if label_ids is not None and label_vocab_file is not None:
                if label_ids != file_label_ids:
                    raise_not_equal_labels_error(
                        first_labels=label_ids,
                        second_labels=file_label_ids,
                        first_labels_desc=f"Labels passed in config parameter "
                        f"`model.common_dataset_parameters.{label_ids_name}`",
                        second_labels_desc=f"Labels loaded from file {plvf} passed in config "
                        f"parameter `model.common_dataset_parameters.{label_vocab_name}",
                    )
            if already_set_label_ids is not None:
                config_label_ids = label_ids if label_vocab_file is None else file_label_ids
                if config_label_ids is not None:
                    if label_vocab_file is None:
                        config_label_ids_source = (
                            f"Labels passed in config parameter `model.common_dataset_parameters.{label_ids_name}`"
                        )
                    else:
                        config_label_ids_source = (
                            f"Labels loaded from file {plvf} passed in config parameter "
                            f"`model.common_dataset_parameters.{label_vocab_name}`"
                        )
                    if already_set_label_ids != config_label_ids:
                        raise_not_equal_labels_error(
                            first_labels=config_label_ids,
                            second_labels=already_set_label_ids,
                            first_labels_desc=config_label_ids_source,
                            second_labels_desc=f"Labels which are already set in an attribute "
                            f"`PunctuationCapitalizationModel.{label_ids_name}`",
                        )
        if plvf is not None:
            pli = load_label_ids(plvf)
        if clvf is not None:
            cli = load_label_ids(clvf)
        for label_ids, parameter_name in [
            (pli, 'punct_label_vocab_file' if pli is None else 'punct_label_ids'),
            (cli, 'capit_label_vocab_file' if cli is None else 'capit_label_ids'),
        ]:
            if label_ids is not None and label_ids[pad_label] != 0:
                raise ValueError(
                    f"Pad label '{pad_label}' has non zero id {label_ids[pad_label]} in "
                    f"`model.common_dataset_parameters.{parameter_name}`."
                )

    def _extract_label_vocab_files_from_config(self) -> Tuple[Optional[Path], Optional[Path]]:
        if self._is_model_being_restored():
            punct_label_vocab_file = self._cfg.class_labels.punct_labels_file
            capit_label_vocab_file = self._cfg.class_labels.capit_labels_file
        else:
            if self._cfg.common_dataset_parameters.label_vocab_dir is None:
                punct_label_vocab_file, capit_label_vocab_file = None, None
            else:
                label_vocab_dir = Path(self._cfg.common_dataset_parameters.label_vocab_dir).expanduser()
                punct_label_vocab_file = label_vocab_dir / self._cfg.class_labels.punct_labels_file
                capit_label_vocab_file = label_vocab_dir / self._cfg.class_labels.capit_labels_file
        return punct_label_vocab_file, capit_label_vocab_file

    def _set_label_ids(self) -> None:
        """
        Set model attributes ``punct_label_ids`` and ``capit_label_ids`` based on label ids passed in config
        item ``common_dataset_parameters``.

        This method also registers artifacts ``class_labels.punct_labels_file`` and ``class_labels.capit_labels_file``.

        This method is called if you do not plan to infer label ids from training file with labels. If training file
        with labels is going to be used, then calling :meth:`~setup_training_data` is enough to set
        ``punct_label_ids`` and ``capit_label_ids`` and register label artifacts.
        """
        punct_label_vocab_file, capit_label_vocab_file = self._extract_label_vocab_files_from_config()
        if punct_label_vocab_file is not None:
            punct_labels_file = self.register_artifact('class_labels.punct_labels_file', str(punct_label_vocab_file))
            if punct_labels_file is None:
                logging.warning(
                    f"The artifact `class_labels.punct_labels_file` was not found in checkpoint. Will rely on "
                    f"`punct_label_ids` parameter"
                )
                self.punct_label_ids = OmegaConf.to_container(self._cfg.common_dataset_parameters.punct_label_ids)
            else:
                self.punct_label_ids = load_label_ids(
                    self.register_artifact('class_labels.punct_labels_file', str(punct_label_vocab_file))
                )
        elif self._cfg.common_dataset_parameters.punct_label_ids is not None:
            self.punct_label_ids = OmegaConf.to_container(self._cfg.common_dataset_parameters.punct_label_ids)
        else:
            raise ValueError(
                f"Could not set attribute `punct_label_ids`. Config parameters "
                f"`model.common_dataset_parameters.punct_label_ids`, "
                f"`model.common_dataset_parameters.punct_label_vocab_file` are not set. Another way to set "
                f"`punct_label_ids` is calling method `setup_training_data`. That way punctuation label ids will be "
                f"inferred from training set."
            )
        if capit_label_vocab_file is not None:
            capit_labels_file = self.register_artifact('class_labels.capit_labels_file', str(capit_label_vocab_file))
            if capit_labels_file is None:
                logging.warning(
                    f"The artifact `class_labels.capit_labels_file` was not found in checkpoint. Will rely on "
                    f"`capit_label_ids` parameter"
                )
                self.capit_label_ids = OmegaConf.to_container(self._cfg.common_dataset_parameters.capit_label_ids)
            else:
                self.capit_label_ids = load_label_ids(
                    self.register_artifact('class_labels.capit_labels_file', str(capit_label_vocab_file))
                )
        elif self._cfg.common_dataset_parameters.capit_label_ids is not None:
            self.capit_label_ids = OmegaConf.to_container(self._cfg.common_dataset_parameters.capit_label_ids)
        else:
            raise ValueError(
                f"Could not set attribute `capit_label_ids`. Config parameters "
                f"`model.common_dataset_parameters.capit_label_ids`, "
                f"`model.common_dataset_parameters.capit_label_vocab_file` are not set. Another way to set "
                f"`capit_label_ids` is calling method `setup_training_data`. That way capitalization label ids will "
                f"be inferred from training set."
            )
        self.label_ids_are_set = True

    def _setup_dataloader_from_config(self, cfg: DictConfig, train: bool) -> torch.utils.data.DataLoader:
        """
        Creates dataset and data loader according to config ``cfg``. If ``train=False`` and attributes
        ``punct_label_ids`` and ``capit_label_ids`` are not set, then this method sets the attributes and registers
        label artifacts.

        Args:
            cfg (:obj:`DictConfig`): a config which follows dataclass
                :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.PunctuationCapitalizationEvalDataConfig`
                Note that list ``ds_item`` is not supported because list ``ds_item`` is unpacked by NeMo core
                instruments
            train (:obj:`bool`): whether train data is set. If ``True``, then label ids are not set in this function
        """
        self._check_label_config_parameters()
        if not self.label_ids_are_set and not train:
            self._set_label_ids()
        if cfg.use_tarred_dataset:
            if cfg.tar_metadata_file is None:
                raise ValueError(
                    f"If parameter `use_tarred_dataset` is `True`, then a field `tar_metadata_file` has to be a path "
                    f"to tarred dataset metadata file, whereas `None` is given."
                )
            tar_metadata_file = Path(cfg.ds_item) / cfg.tar_metadata_file
            dataset = BertPunctuationCapitalizationTarredDataset(
                metadata_file=tar_metadata_file,
                tokenizer=self.tokenizer,
                pad_label=self._cfg.common_dataset_parameters.pad_label,
                ignore_extra_tokens=self._cfg.common_dataset_parameters.ignore_extra_tokens,
                ignore_start_end=self._cfg.common_dataset_parameters.ignore_start_end,
                world_size=self.world_size,
                global_rank=self.global_rank,
                shuffle_n=cfg.tar_shuffle_n,
                shard_strategy=cfg.shard_strategy,
                label_info_save_dir=cfg.label_info_save_dir,
                use_audio=cfg.use_audio,
            )
            dataset.check_for_label_consistency_with_model_config(
                self.punct_label_ids,
                self.capit_label_ids,
                self._cfg.class_labels,
                self._cfg.common_dataset_parameters,
            )
        else:
            if cfg.text_file is None or cfg.labels_file is None:
                raise ValueError(
                    f"If parameter `use_tarred_dataset` is `False`, then fields `text_file` and `labels_file` in "
                    f"dataset config must not be `None`. Whereas `text_file={cfg.text_file}` and "
                    f"`label_file={cfg.labels_file}`."
                )
            if cfg.tokens_in_batch is None and cfg.use_bucketing:
                raise ValueError(
                    f"If `use_tarred_dataset` is `False`, then you need to provide `tokens_in_batch` parameter."
                )
            text_file, labels_file, = Path(cfg.ds_item) / cfg.text_file, Path(cfg.ds_item) / cfg.labels_file
            if cfg.audio_file:
                audio_file = Path(cfg.ds_item) / cfg.audio_file
            if self.label_ids_are_set:
                label_kwargs = {'punct_label_ids': self.punct_label_ids, 'capit_label_ids': self.capit_label_ids}
            else:
                punct_label_vocab_file, capit_label_vocab_file = self._extract_label_vocab_files_from_config()
                label_kwargs = {
                    'punct_label_ids': self._cfg.common_dataset_parameters.punct_label_ids,
                    'capit_label_ids': self._cfg.common_dataset_parameters.capit_label_ids,
                    'punct_label_vocab_file': punct_label_vocab_file,
                    'capit_label_vocab_file': capit_label_vocab_file,
                }
            if train:
                number_of_batches_is_multiple_of = 1
                if self._trainer is None:
                    warnings.warn(
                        'A model attribute `trainer` is not set before training dataset setting. If training is '
                        'resumed from checkpoint, then current epoch data loading can be distorted: some batches '
                        'may be processed several times and some can be not processed at all. `trainer.current_epoch`'
                        ' is used as random seed for shuffling batches. Now 0 will be used. If the '
                        'checkpoint was created not during initial epoch a shuffling of the dataset will '
                        'be different. You may try use `exp_manager()` function and '
                        '`PunctuationCapitalizationModel.set_trainer()` method before '
                        '`PunctuationCapitalizationModel.setup_training_data()` method.'
                    )
                    batch_shuffling_random_seed = 0
                else:
                    batch_shuffling_random_seed = self._trainer.current_epoch
            else:
                batch_shuffling_random_seed = 0
                if self._trainer is None:
                    warnings.warn(
                        'A model attribute `trainer` is not set before test or validation dataset setting. If more '
                        'than 1 GPU is used for testing, then some examples may be tested several times because '
                        'number of batches may be not evenly divisible by number of processes. This leads to '
                        'distortion of metrics. See more in description of `number_of_batches_is_multiple_of` '
                        'parameter of class `BertPunctuationCapitalizationDataset` initializer and '
                        'https://pytorch.org/docs/stable/data.html#multi-process-data-loading. You may try to use '
                        '`PunctuationCapitalizationModel.set_trainer()` method before '
                        '`PunctuationCapitalizationModel.setup_validation_data()` and '
                        '`PunctuationCapitalizationModel.setup_test_data()` methods.'
                    )
                    number_of_batches_is_multiple_of = 1
                else:
                    number_of_batches_is_multiple_of = self._trainer.num_nodes * self._trainer.num_devices
            if cfg.cache_dir is None:
                cache_dir = cfg.cache_dir
            else:
                # If pickled features are saved `cache_dir` not in the same directory with original data files, then
                # a full path to data directory have to be appended to `cache_dir`. This is done to avoid collisions
                # cache for different datasets is saved to same `cache_dir`.
                cache_dir = Path(cfg.cache_dir).joinpath('fsroot', *text_file.expanduser().resolve().parts[1:-1])
            dataset = BertPunctuationCapitalizationDataset(
                tokenizer=self.tokenizer,
                text_file=text_file,
                labels_file=labels_file,
                pad_label=self._cfg.common_dataset_parameters.pad_label,
                **label_kwargs,
                max_seq_length=cfg.max_seq_length,
                ignore_extra_tokens=self._cfg.common_dataset_parameters.ignore_extra_tokens,
                ignore_start_end=self._cfg.common_dataset_parameters.ignore_start_end,
                use_cache=cfg.use_cache,
                num_samples=cfg.num_samples,
                tokens_in_batch=cfg.tokens_in_batch,
                n_jobs=cfg.n_jobs,
                number_of_batches_is_multiple_of=number_of_batches_is_multiple_of,
                batch_shuffling_random_seed=batch_shuffling_random_seed,
                verbose=cfg.verbose,
                get_label_frequencies=cfg.get_label_frequences,
                cache_dir=cache_dir,
                label_info_save_dir=cfg.label_info_save_dir,
                audio_file=audio_file if cfg.audio_file else None,
                sample_rate=cfg.sample_rate,
                use_audio=cfg.use_audio,
                use_bucketing=cfg.use_bucketing,
                preload_audios=cfg.preload_audios,
            )
        if cfg.shuffle and cfg.use_tarred_dataset:
            logging.warning(f"Shuffling in dataloader is not supported for tarred dataset.")
            shuffle = False
        else:
            shuffle = cfg.shuffle
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=1 if cfg.use_bucketing else cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        )

    def _setup_infer_dataloader(
        self,
        queries: List[str],
        batch_size: int,
        max_seq_length: int,
        step: int,
        margin: int,
        dataloader_kwargs: Optional[Dict[str, Any]],
        audio_queries: Optional[Union[List[bytes], List[str]]] = None,
        target_sr: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        """
        Setup function for an infer data loader.

        Args:
            queries (:obj:`List[str]`): lower cased text without punctuation
            batch_size (:obj:`int`): batch size to use during inference
            max_seq_length (:obj:`int`): length of segments into which queries are split. ``max_seq_length`` includes
                ``[CLS]`` and ``[SEP]`` so every segment contains at most ``max_seq_length-2`` tokens from input a
                query.
            step (:obj:`int`): number of tokens by which a segment is offset to a previous segment. Parameter ``step``
                cannot be greater than ``max_seq_length-2``.
            margin (:obj:`int`): number of tokens near the edge of a segment which label probabilities are not used in
                final prediction computation.
            audio_queries (:obj:`List[str]`, `optional`): paths to audio files.
            target_sr (:obj:`int`, `optional`): target sample rate for audios.
        Returns:
            :obj:`torch.utils.data.DataLoader`: inference data loader
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        dataset = BertPunctuationCapitalizationInferDataset(
            tokenizer=self.tokenizer,
            queries=queries,
            max_seq_length=max_seq_length,
            step=step,
            margin=margin,
            audio_queries=audio_queries,
            target_sr=target_sr,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **dataloader_kwargs,
        )

    @staticmethod
    def _remove_margins(tensor: torch.Tensor, margin_size: int, keep_left: bool, keep_right: bool) -> torch.Tensor:
        tensor = tensor.detach().clone()
        if not keep_left:
            tensor = tensor[margin_size + 1 :]  # remove left margin and CLS token
        if not keep_right:
            tensor = tensor[: tensor.shape[0] - margin_size - 1]  # remove right margin and SEP token
        return tensor

    def _transform_logit_to_prob_and_remove_margins_and_extract_word_probs(
        self,
        punct_logits: torch.Tensor,
        capit_logits: torch.Tensor,
        subtokens_mask: torch.Tensor,
        start_word_ids: Tuple[int],
        margin: int,
        is_first: Tuple[bool],
        is_last: Tuple[bool],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Applies softmax to get punctuation and capitalization probabilities, applies ``subtokens_mask`` to extract
        probabilities for words from probabilities for tokens, removes ``margin`` probabilities near edges of a segment.
        Left margin of the first segment in a query and right margin of the last segment in a query are not removed.
        Calculates new ``start_word_ids`` taking into the account the margins. If the left margin of a segment is
        removed corresponding start word index is increased by number of words (number of nonzero values in
        corresponding ``subtokens_mask``) in the margin.

        Args:
            punct_logits: a float tensor of shape ``[batch_size, segment_length, number_of_punctuation_labels]``
            capit_logits: a float tensor of shape ``[batch_size, segment_length, number_of_capitalization_labels]``
            subtokens_mask: a float tensor of shape ``[batch_size, segment_length]``
            start_word_ids: indices of segment first words in a query
            margin: number of tokens near edges of a segment which probabilities are discarded
            is_first: is segment the first segment in a query
            is_last: is segment the last segment in a query
        Returns:
            b_punct_probs: list containing ``batch_size`` numpy arrays. The numpy arrays have shapes
                ``[number_of_word_in_this_segment, number_of_punctuation_labels]``. Word punctuation probabilities for
                segments in the batch.
            b_capit_probs: list containing ``batch_size`` numpy arrays. The numpy arrays have shapes
                ``[number_of_word_in_this_segment, number_of_capitalization_labels]``. Word capitalization
                probabilities for segments in the batch.
            new_start_word_ids: indices of segment first words in a query after margin removal
        """
        new_start_word_ids = list(start_word_ids)
        subtokens_mask = subtokens_mask > 0.5
        b_punct_probs, b_capit_probs = [], []
        for i, (first, last, pl, cl, stm) in enumerate(
            zip(is_first, is_last, punct_logits, capit_logits, subtokens_mask)
        ):
            if not first:
                new_start_word_ids[i] += torch.count_nonzero(stm[: margin + 1]).numpy()  # + 1 is for [CLS] token
            stm = self._remove_margins(stm, margin, keep_left=first, keep_right=last)
            for b_probs, logits in [(b_punct_probs, pl), (b_capit_probs, cl)]:
                p = torch.nn.functional.softmax(
                    self._remove_margins(logits, margin, keep_left=first, keep_right=last)[stm], dim=-1,
                )
                b_probs.append(p.detach().cpu().numpy())
        return b_punct_probs, b_capit_probs, new_start_word_ids

    @staticmethod
    def _move_acc_probs_to_token_preds(
        pred: List[int], acc_prob: np.ndarray, number_of_probs_to_move: int
    ) -> Tuple[List[int], np.ndarray]:
        """
        ``number_of_probs_to_move`` rows in the beginning are removed from ``acc_prob``. From every remove row the label
        with the largest probability is selected and appended to ``pred``.
        Args:
            pred: list with ready label indices for a query
            acc_prob: numpy array of shape ``[number_of_words_for_which_probabilities_are_accumulated, number_of_labels]``
            number_of_probs_to_move: int
        Returns:
            pred: list with ready label indices for a query
            acc_prob: numpy array of shape
                ``[number_of_words_for_which_probabilities_are_accumulated - number_of_probs_to_move, number_of_labels]``
        """
        if number_of_probs_to_move > acc_prob.shape[0]:
            raise ValueError(
                f"Not enough accumulated probabilities. Number_of_probs_to_move={number_of_probs_to_move} "
                f"acc_prob.shape={acc_prob.shape}"
            )
        if number_of_probs_to_move > 0:
            pred = pred + list(np.argmax(acc_prob[:number_of_probs_to_move], axis=-1))
        acc_prob = acc_prob[number_of_probs_to_move:]
        return pred, acc_prob

    @staticmethod
    def _update_accumulated_probabilities(acc_prob: np.ndarray, update: np.ndarray) -> np.ndarray:
        """
        Args:
            acc_prob: numpy array of shape ``[A, L]``
            update: numpy array of shape ``[A + N, L]``
        Returns:
            numpy array of shape ``[A + N, L]``
        """
        acc_prob = np.concatenate([acc_prob * update[: acc_prob.shape[0]], update[acc_prob.shape[0] :]], axis=0)
        return acc_prob

    def _apply_punct_capit_predictions(self, query: str, punct_preds: List[int], capit_preds: List[int]) -> str:
        """
        Restores punctuation and capitalization in ``query``.
        Args:
            query: a string without punctuation and capitalization
            punct_preds: ids of predicted punctuation labels
            capit_preds: ids of predicted capitalization labels
        Returns:
            a query with restored punctuation and capitalization
        """
        query = query.strip().split()
        assert len(query) == len(
            punct_preds
        ), f"len(query)={len(query)} len(punct_preds)={len(punct_preds)}, query[:30]={query[:30]}"
        assert len(query) == len(
            capit_preds
        ), f"len(query)={len(query)} len(capit_preds)={len(capit_preds)}, query[:30]={query[:30]}"
        punct_ids_to_labels = {v: k for k, v in self.punct_label_ids.items()}
        capit_ids_to_labels = {v: k for k, v in self.capit_label_ids.items()}
        query_with_punct_and_capit = ''
        for j, word in enumerate(query):
            punct_label = punct_ids_to_labels[punct_preds[j]]
            capit_label = capit_ids_to_labels[capit_preds[j]]

            if capit_label != self._cfg.common_dataset_parameters.pad_label:
                word = word.capitalize()
            query_with_punct_and_capit += word
            if punct_label != self._cfg.common_dataset_parameters.pad_label:
                query_with_punct_and_capit += punct_label
            query_with_punct_and_capit += ' '
        return query_with_punct_and_capit[:-1]

    def _get_labels(self, punct_preds: List[int], capit_preds: List[int]) -> str:
        """
        Returns punctuation and capitalization labels in NeMo format for encoded punctuation ``punct_preds``
        and ``capit_preds`` labels (see https://docs.nvidia.com/deeplearning/nemo/
        user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format).
        Args:
            punct_preds: ids of predicted punctuation labels
            capit_preds: ids of predicted capitalization labels
        Returns:
            labels in NeMo format
        """
        assert len(capit_preds) == len(
            punct_preds
        ), f"len(capit_preds)={len(capit_preds)} len(punct_preds)={len(punct_preds)}"
        punct_ids_to_labels = {v: k for k, v in self.punct_label_ids.items()}
        capit_ids_to_labels = {v: k for k, v in self.capit_label_ids.items()}
        result = ''
        for capit_label, punct_label in zip(capit_preds, punct_preds):
            punct_label = punct_ids_to_labels[punct_label]
            capit_label = capit_ids_to_labels[capit_label]
            result += punct_label + capit_label + ' '
        return result[:-1]

    def add_punctuation_capitalization(
        self,
        queries: List[str],
        batch_size: int = None,
        max_seq_length: int = 64,
        step: int = 8,
        margin: int = 16,
        return_labels: bool = False,
        dataloader_kwargs: Dict[str, Any] = None,
    ) -> List[str]:
        """
        Adds punctuation and capitalization to the queries. Use this method for inference.

        Parameters ``max_seq_length``, ``step``, ``margin`` are for controlling the way queries are split into segments
        which are processed by the model. Parameter ``max_seq_length`` is a length of a segment after tokenization
        including special tokens [CLS] in the beginning and [SEP] in the end of a segment. Parameter ``step`` is a
        shift between consequent segments. Parameter ``margin`` is used to exclude negative effect of subtokens near
        borders of segments which have only one side context.

        If segments overlap, probabilities of overlapping predictions are multiplied and then the label with
        corresponding to the maximum probability is selected.

        Args:
            queries (:obj:`List[str]`): lower cased text without punctuation.
            batch_size (:obj:`List[str]`, `optional`): batch size to use during inference. If ``batch_size`` parameter
                is not provided, then it will be equal to length of ``queries`` list.
            max_seq_length (:obj:`int`, `optional`, defaults to :obj:`64`): maximum sequence length of a segment after
                tokenization including :code:`[CLS]` and :code:`[SEP]` tokens.
            step (:obj:`int`, `optional`, defaults to :obj:`8`): relative shift of consequent segments into which long
                queries are split. Long queries are split into segments which can overlap. Parameter ``step`` controls
                such overlapping. Imagine that queries are tokenized into characters, ``max_seq_length=5``, and
                ``step=2``. In such case, query ``"hello"`` is tokenized into segments
                ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]``.
            margin (:obj:`int`, `optional`, defaults to :obj:`16`): number of subtokens in the beginning and the end of
                segments which are not used for prediction computation. The first segment does not have left margin and
                the last segment does not have right margin. For example, if an input sequence is tokenized into
                characters, ``max_seq_length=5``, ``step=1``, and ``margin=1``, then query ``"hello"`` will be
                tokenized into segments ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'e', 'l', 'l', '[SEP]'],
                ['[CLS]', 'l', 'l', 'o', '[SEP]']]``. These segments are passed to the model. Before final predictions
                computation, margins are removed. In the next list, subtokens which logits are not used for final
                predictions computation are marked with asterisk: ``[['[CLS]'*, 'h', 'e', 'l'*, '[SEP]'*],
                ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]``.
            return_labels (:obj:`bool`, `optional`, defaults to :obj:`False`): whether to return labels in NeMo format
                (see :ref:`nemo-data-format-label`) instead of queries with restored
                punctuation and capitalization.
            dataloader_kwargs (:obj:`Dict[str, Any]`, `optional`): an optional dictionary with parameters of PyTorch
                data loader. May include keys: ``'num_workers'``, ``'pin_memory'``, ``'worker_init_fn'``,
                ``'prefetch_factor'``, ``'persistent_workers'``.
        Returns:
            :obj:`List[str]`: a list of queries with restored capitalization and punctuation if
            ``return_labels=False``, else a list of punctuation and capitalization labels strings for all queries
        """
        if len(queries) == 0:
            return []
        if batch_size is None:
            batch_size = len(queries)
            logging.info(f'Using batch size {batch_size} for inference')
        result: List[str] = []
        mode = self.training
        try:
            self.eval()
            infer_datalayer = self._setup_infer_dataloader(
                queries, batch_size, max_seq_length, step, margin, dataloader_kwargs
            )
            # Predicted labels for queries. List of labels for every query
            all_punct_preds: List[List[int]] = [[] for _ in queries]
            all_capit_preds: List[List[int]] = [[] for _ in queries]
            # Accumulated probabilities (or product of probabilities acquired from different segments) of punctuation
            # and capitalization. Probabilities for words in a query are extracted using `subtokens_mask`. Probabilities
            # for newly processed words are appended to the accumulated probabilities. If probabilities for a word are
            # already present in `acc_probs`, old probabilities are replaced with a product of old probabilities
            # and probabilities acquired from new segment. Segments are processed in an order they appear in an
            # input query. When all segments with a word are processed, a label with the highest probability
            # (or product of probabilities) is chosen and appended to an appropriate list in `all_preds`. After adding
            # prediction to `all_preds`, probabilities for a word are removed from `acc_probs`.
            acc_punct_probs: List[Optional[np.ndarray]] = [None for _ in queries]
            acc_capit_probs: List[Optional[np.ndarray]] = [None for _ in queries]
            d = self.device
            for batch_i, batch in tqdm(
                enumerate(infer_datalayer), total=ceil(len(infer_datalayer.dataset) / batch_size), unit="batch"
            ):
                inp_ids, inp_type_ids, inp_mask, subtokens_mask, start_word_ids, query_ids, is_first, is_last = batch
                punct_logits, capit_logits = self.forward(
                    input_ids=inp_ids.to(d), token_type_ids=inp_type_ids.to(d), attention_mask=inp_mask.to(d),
                )
                _res = self._transform_logit_to_prob_and_remove_margins_and_extract_word_probs(
                    punct_logits, capit_logits, subtokens_mask, start_word_ids, margin, is_first, is_last
                )
                punct_probs, capit_probs, start_word_ids = _res
                for i, (q_i, start_word_id, bpp_i, bcp_i) in enumerate(
                    zip(query_ids, start_word_ids, punct_probs, capit_probs)
                ):
                    for all_preds, acc_probs, b_probs_i in [
                        (all_punct_preds, acc_punct_probs, bpp_i),
                        (all_capit_preds, acc_capit_probs, bcp_i),
                    ]:
                        if acc_probs[q_i] is None:
                            acc_probs[q_i] = b_probs_i
                        else:
                            all_preds[q_i], acc_probs[q_i] = self._move_acc_probs_to_token_preds(
                                all_preds[q_i], acc_probs[q_i], start_word_id - len(all_preds[q_i]),
                            )
                            acc_probs[q_i] = self._update_accumulated_probabilities(acc_probs[q_i], b_probs_i)
            for all_preds, acc_probs in [(all_punct_preds, acc_punct_probs), (all_capit_preds, acc_capit_probs)]:
                for q_i, (pred, prob) in enumerate(zip(all_preds, acc_probs)):
                    if prob is not None:
                        all_preds[q_i], acc_probs[q_i] = self._move_acc_probs_to_token_preds(pred, prob, len(prob))
            for i, query in enumerate(queries):
                result.append(
                    self._get_labels(all_punct_preds[i], all_capit_preds[i])
                    if return_labels
                    else self._apply_punct_capit_predictions(query, all_punct_preds[i], all_capit_preds[i])
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
            :obj:`List[PretrainedModelInfo]`: a list of available pre-trained models.
        """
        result = [
            PretrainedModelInfo(
                pretrained_model_name="punctuation_en_bert",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_bert/versions/1.0.0rc1/"
                "files/punctuation_en_bert.nemo",
                description="The model was trained with NeMo BERT base uncased checkpoint on a subset of data from "
                "the following sources: Tatoeba sentences, books from Project Gutenberg, Fisher transcripts.",
            ),
            PretrainedModelInfo(
                pretrained_model_name="punctuation_en_distilbert",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_distilbert/versions/"
                "1.0.0rc1/files/punctuation_en_distilbert.nemo",
                description="The model was trained with DistilBERT base uncased checkpoint from HuggingFace on a "
                "subset of data from the following sources: Tatoeba sentences, books from Project Gutenberg, "
                "Fisher transcripts.",
            ),
        ]
        return result

    @property
    def output_module(self):
        return self
