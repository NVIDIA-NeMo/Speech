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

# pylint: skip-file
# flake8: noqa

import json
from functools import partial

import torch
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import DictConfig, ListConfig

from nemo.collections.common.data import ConcatMapDataset
from nemo.collections.common.metrics import MetricStringToTorchMetric
from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric

try:
    from nemo.collections.nlp.data.common.sequence_to_sequence_dataset import SequenceToSequenceDataset
except ModuleNotFoundError:
    from abc import ABC

    SequenceToSequenceDataset = ABC

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.retro_fine_tune_dataset import RetroQAFineTuneDataset
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model, T5Sentinel
from nemo.collections.nlp.parts.nlp_overrides import GlobalBatchDataFetcher
from nemo.utils import AppState, logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


__all__ = ['MegatronRetroFinetuneModel']


def build_all_datasets(
    cfg,
    tokenizer,
    train_valid_test_num_samples,
):
    """Build train, valid, and test RETRO datasets.
    There is one to one mapping between data_prefix and knn_map_path.
    Currently only supports one retrieval dataset.
    """
    train_dataset = RetroQAFineTuneDataset(
        cfg.train_ds.get('file_name'),
        tokenizer,
        cfg.train_ds.get('answer_only_loss'),
        tokenizer.pad_id,
        cfg.train_ds.get('seq_length'),
        cfg.train_ds.get('add_bos'),
        cfg.train_ds.get('add_eos'),
        train_valid_test_num_samples[0],
        cfg.train_ds.get('seed'),
        cfg.train_ds.get('neighbors'),
    )
    val_dataset = RetroQAFineTuneDataset(
        cfg.val_ds.get('file_name'),
        tokenizer,
        cfg.val_ds.get('answer_only_loss'),
        tokenizer.pad_id,
        cfg.val_ds.get('seq_length'),
        cfg.val_ds.get('add_bos'),
        cfg.val_ds.get('add_eos'),
        train_valid_test_num_samples[1],
        cfg.val_ds.get('seed'),
        cfg.val_ds.get('neighbors'),
    )
    test_dataset = RetroQAFineTuneDataset(
        cfg.test_ds.get('file_name'),
        tokenizer,
        cfg.test_ds.get('answer_only_loss'),
        tokenizer.pad_id,
        cfg.test_ds.get('seq_length'),
        cfg.test_ds.get('add_bos'),
        cfg.test_ds.get('add_eos'),
        train_valid_test_num_samples[2],
        cfg.test_ds.get('seed'),
        cfg.test_ds.get('neighbors'),
    )

    return train_dataset, val_dataset, test_dataset


class MegatronRetroFinetuneModel(MegatronRetrievalModel):
    """Finetune RETRO Model"""

    def build_train_valid_test_datasets(self):
        logging.info('Building RETRO datasets.')
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size // self.cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = int(self.trainer.limit_test_batches)

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        self._train_ds, self._validation_ds, self._test_ds = build_all_datasets(
            cfg=self.cfg.data,
            tokenizer=self.tokenizer,
            train_valid_test_num_samples=train_valid_test_num_samples,
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building RETRO datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        if isinstance(dataset, BlendableDataset):
            collate_fun = dataset.datasets[0].collate_fn
        else:
            collate_fun = dataset.collate_fn

        collate_fn = partial(collate_fun, tp_workers=0)
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size // self.cfg.tensor_model_parallel_size
        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.cfg.micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
