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

"""RETRO style dataset."""

import os
import time

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import deallocate_indexed_dataset_memory
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset as make_indexed_dataset
from nemo.core import Dataset
from nemo.utils import logging

try:
    from megatron.core import mpu, tensor_parallel
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.retro.config import RetroGPTChunkDatasets
    from megatron.core.datasets.retro.query.multi_split_gpt_dataset import (
        MultiSplitGPTDataset,
        MultiSplitGPTDatasetConfig,
    )
    from megatron.core.datasets.retro.query.retro_dataset import get_retro_datasets
    from megatron.core.datasets.utils import get_blend_from_list
    from megatron.core.models.retro import RetroConfig

    from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class RETRODataset(Dataset):
    def __init__(self, cfg, retro_config: RetroConfig, tokenizer, mcore_retro_dataset, number_samples_with_neighbors):
        super().__init__()

        self.reset_position_ids = cfg.data.get('reset_position_ids', False)
        self.reset_attention_mask = cfg.data.get('reset_attention_mask', False)
        self.eod_mask_loss = cfg.data.get('eod_mask_loss', False)
        self.eos_id = tokenizer.eos_id
        self.retro_config = retro_config
        self.mcore_retro_dataset = mcore_retro_dataset
        self.number_samples_with_neighbors = number_samples_with_neighbors  # quick fix for problems of mismatch in processed/indexed retro data, # of GPT samples is different from # of samples with neighbors retrieved
        self.tokenizer = tokenizer

        return

    def __len__(self):
        return len(self.mcore_retro_dataset.chunk_dataset.sample_dataset)

    def _get_text(self, idx: int):
        # return the tokens ids of idx
        # Caveat: these tokens are got from the already pre-tokenized data file, mcore's GPTDataset doesn't run __getitem__, only run _query_document_sample_shuffle_indices
        return self.mcore_retro_dataset[idx]

    def __getitem__(self, idx):

        # quick fix for problems of mismatch in processed/indexed retro data, # of GPT samples is different from # of samples with neighbors retrieved
        idx = idx % self.number_samples_with_neighbors

        sample = self._get_text(idx)

        # Unpack
        tokens_ = torch.from_numpy(sample['text'])
        tokens_ = tokens_.long()  # size should be [seq_length]
        labels = tokens_[1:].contiguous()
        tokens = tokens_[:-1].contiguous()
        neighbor_tokens = torch.from_numpy(sample['neighbor_tokens'])
        neighbor_tokens = neighbor_tokens.long()  # size should be [l, k, r]

        # note: [l, k, r]  => [l*k, r]
        # note: 2x == neighbor, continuation
        neighbor_tokens = neighbor_tokens.view(-1, self.retro_config.retro_retrieved_length).long()

        # Get the masks and postition ids for tokens and neighbor_tokens
        tokens = torch.unsqueeze(
            tokens, 0
        )  # get_ltor_masks_and_position_ids takes as input tokens arguments as a batch (2D tensor), so need to convert tokens from 1D to 2D
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens, self.eos_id, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss
        )
        tokens, attention_mask, loss_mask, position_ids = tokens[0], attention_mask[0], loss_mask[0], position_ids[0]
        _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(  # neighbor_tokens is already a 2D array
            neighbor_tokens, self.eos_id, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss
        )
        neighbor_attention_mask = torch.zeros(
            [1, 1]
        )  # just a dummy values, since the batch neighbor_attention_mask will be set to None in megatron_retro_model.py following Lawrence's implementation

        return {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'context_input_ids': neighbor_tokens,
            'context_attention_mask': neighbor_attention_mask,
            'context_position_ids': neighbor_position_ids,
        }


def build_train_valid_test_datasets(
    cfg, retro_config: RetroConfig, train_valid_test_num_samples, seq_length, tokenizer,
):

    # gpt dataset
    train_ds, valid_ds, test_ds = gpt_train_valid_test_datasets_provider(cfg, train_valid_test_num_samples, tokenizer)

    gpt_datasets = {
        "train": (train_ds, train_valid_test_num_samples[0]),
        "valid": (valid_ds, train_valid_test_num_samples[1]),
        "test": (test_ds, train_valid_test_num_samples[2]),
    }

    retro_train_ds, retro_valid_ds, retro_test_ds = get_retro_datasets(
        config=retro_config, gpt_datasets=gpt_datasets, sample_length=seq_length, eod_token_id=tokenizer.eos_id,
    )

    train_ds = (
        RETRODataset(
            cfg=cfg,
            retro_config=retro_config,
            tokenizer=tokenizer,
            mcore_retro_dataset=retro_train_ds,
            number_samples_with_neighbors=train_valid_test_num_samples[0],
        )
        if retro_train_ds
        else None
    )
    valid_ds = (
        RETRODataset(
            cfg=cfg,
            retro_config=retro_config,
            tokenizer=tokenizer,
            mcore_retro_dataset=retro_valid_ds,
            number_samples_with_neighbors=train_valid_test_num_samples[1],
        )
        if retro_valid_ds
        else None
    )
    test_ds = (
        RETRODataset(
            cfg=cfg,
            retro_config=retro_config,
            tokenizer=tokenizer,
            mcore_retro_dataset=retro_test_ds,
            number_samples_with_neighbors=train_valid_test_num_samples[2],
        )
        if retro_test_ds
        else None
    )

    return train_ds, valid_ds, test_ds


def gpt_train_valid_test_datasets_provider(cfg, train_val_test_num_samples, tokenizer):
    """Build the train test and validation datasets.
       Implemented from train_valid_test_datasets_provider in M-LM/pretrain_gpt.py

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """

    def is_dataset_built_on_rank():
        return (
            mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
        ) and mpu.get_tensor_model_parallel_rank() == 0

    data_config = MultiSplitGPTDatasetConfig(
        random_seed=cfg.seed,
        sequence_length=cfg.data.seq_length,
        blend=get_blend_from_list(cfg.data.data_prefix),
        split=cfg.data.splits_string,
        split_preprocessing=cfg.data.retro_data.retro_split_preprocessing,
        path_to_cache=None,
        return_document_ids=False,
        reset_position_ids=cfg.data.get('reset_position_ids', False),
        reset_attention_mask=cfg.data.get('reset_attention_mask', False),
        eod_mask_loss=cfg.data.get('eod_mask_loss', False),
        tokenizer=tokenizer,
    )

    print("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MultiSplitGPTDataset, train_val_test_num_samples, is_dataset_built_on_rank, data_config
    ).build()

    print("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds
