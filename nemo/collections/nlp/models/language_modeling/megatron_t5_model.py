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

import enum
import math

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)
from nemo.utils import logging

__all__ = ["MegatronT5Model"]


class T5Sentinel(enum.Enum):
    FIRST = '<extra_id_0>'
    END = '<extra_id_1>'


class MegatronT5Model(MegatronLMEncoderDecoderModel):
    """
    Megatron T5 pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        # validate cfg
        self._validate_cfg()

    @property
    def model_name(self):
        """Allows child classes to implement models with different data regime"""
        return "T5"

    def _validate_cfg(self):
        """Class-specific cfg validation"""
        # Make sure the user specifies dataset type as either 't5' or 't5_prefix_lm' only.
        if self._cfg.data.get('dataset_type', None) is not None:
            if self._cfg.data.get('dataset_type') not in ['t5', 't5_prefix_lm', 'ul2']:
                raise ValueError(
                    f"dataset_type must be either 't5', 't5_prefix_lm' or 'ul2'. found {self._cfg.data.get('dataset_type')}"
                )

        if hasattr(self._cfg.data, 'seq_length_dec') and self._cfg.data.get('dataset_type') == 't5':
            if self._cfg.data.seq_length_dec < self._cfg.data.seq_length * self._cfg.data.masked_lm_prob:
                raise ValueError(
                    f"Cannot have decoder max sequence length ({self._cfg.data.seq_length_dec}) less than encoder sequence length ({self._cfg.data.seq_length}) * masked_lm_prob ({self._cfg.data.masked_lm_prob})"
                )

        if self._cfg.data.get("dataset_type", "t5") == "ul2":
            if self._cfg.data.seq_length_dec != self._cfg.data.seq_length:
                raise ValueError(
                    f"Encoder and decoder sequence lengths must be the same while using the UL2 dataset type. Found encoder length {self._cfg.data.seq_length} and decoder length {self._cfg.data.seq_length_dec}"
                )
            if (
                self._cfg.tokenizer.num_sentinel_tokens
                < self._cfg.data.seq_length * self._cfg.data.extreme_masked_lm_prob
            ):
                raise ValueError(
                    f"Not enough sentinel tokens specified. Need at least {math.ceil(self._cfg.data.seq_length * self._cfg.data.extreme_masked_lm_prob)} sentinel tokens. Found {self._cfg.tokenizer.num_sentinel_tokens}"
                )

    @property
    def _build_train_valid_test_datasets_kwargs(self):
        """allows child classes to add kwargs to dataset building"""
        return dict(max_seq_length_dec=self._cfg.data.seq_length_dec,)

    def _build_vocab(self):
        self.num_sentinel_tokens = self._cfg.tokenizer.num_sentinel_tokens
        MegatronT5Model.add_special_tokens_to_tokenizer(
            tokenizer=self.tokenizer,
            tokenizer_cfg=self._cfg.tokenizer,
            dataset_type=self._cfg.data.get("dataset_type", "t5"),
            add_sentinel_tokens_in_reverse_order=self._cfg.tokenizer.get(
                "add_sentinel_tokens_in_reverse_order", False
            ),
            add_sentinel_tokens_first=self._cfg.tokenizer.get("add_sentinel_tokens_first", False),
        )
        super()._build_vocab()

    @classmethod
    def _add_sentinel_tokens(cls, tokenizer, num_sentinel_tokens, add_sentinel_tokens_in_reverse_order):
        # Special check to see if <extra_id_{}> is already present in the tokenizer. If it is, only modify the additional_special_tokens function.
        for i in range(num_sentinel_tokens):
            if add_sentinel_tokens_in_reverse_order:
                i = num_sentinel_tokens - i - 1
            if len(tokenizer.text_to_ids(f'<extra_id_{i}>')) == 1:
                tokenizer.special_token_to_id[f'<extra_id_{i}>'] = tokenizer.text_to_ids(f'<extra_id_{i}>')[0]
            else:
                tokenizer.add_special_tokens([f'<extra_id_{i}>'])

    @classmethod
    def _add_base_special_tokens(cls, tokenizer, is_huggingface_converted_model):
        # Need to add cls, sep, mask tokens to the tokenizer if they don't exist.
        # If cls, sep and mask are not attributes of the tokenizer, add it.
        if not hasattr(tokenizer, 'cls_token'):
            tokenizer.add_special_tokens({'cls_token': '<cls>'})
        if not hasattr(tokenizer.tokenizer, 'sep_id'):
            tokenizer.add_special_tokens({'sep_token': '<sep>'})
        if not hasattr(tokenizer.tokenizer, 'mask_id'):
            tokenizer.add_special_tokens({'mask_token': '<mask>'})

        # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
        if not hasattr(tokenizer, 'pad_token'):
            # TODO: Figure out how to do backward compat with pad_id > 0 and >= 0.
            if is_huggingface_converted_model:
                if hasattr(tokenizer.tokenizer, 'pad_id') and tokenizer.tokenizer.pad_id() >= 0:
                    tokenizer.pad_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.pad_id())
                else:
                    tokenizer.add_special_tokens({'pad_token': '<pad>'})
            else:
                if hasattr(tokenizer.tokenizer, 'pad_id') and tokenizer.tokenizer.pad_id() > 0:
                    tokenizer.pad_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.pad_id())
                else:
                    tokenizer.add_special_tokens({'pad_token': '<pad>'})
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

        if not hasattr(tokenizer, 'bos_token'):
            if hasattr(tokenizer.tokenizer, 'bos_id') and tokenizer.tokenizer.bos_id() > 0:
                tokenizer.bos_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.bos_id())
            else:
                tokenizer.add_special_tokens({'bos_token': '<bos>'})
        else:
            tokenizer.add_special_tokens({'bos_token': '<s>'})

        if not hasattr(tokenizer, 'eos_token'):
            if hasattr(tokenizer.tokenizer, 'eos_id') and tokenizer.tokenizer.eos_id() > 0:
                tokenizer.eos_token = tokenizer.tokenizer.id_to_piece(tokenizer.tokenizer.eos_id())
            else:
                tokenizer.add_special_tokens({'eos_token': '<eos>'})
        else:
            tokenizer.add_special_tokens({'eos_token': '</s>'})

    @classmethod
    def add_special_tokens_to_tokenizer(
        cls,
        tokenizer,
        tokenizer_cfg,
        dataset_type="t5",
        add_sentinel_tokens_in_reverse_order=False,
        add_sentinel_tokens_first=False,
    ):
        # T5-related construction
        if tokenizer_cfg.library == 'huggingface' or tokenizer_cfg.library == 'megatron':
            additional_tokens = {
                'additional_special_tokens': [
                    f'<extra_id_{i}>' for i in range(tokenizer_cfg.get('num_sentinel_tokens', 0))
                ]
            }
            if dataset_type == "ul2":
                mask_types = ['r', 's', 'x']
                for mask_type in mask_types:
                    additional_tokens['additional_special_tokens'].extend([f'<extra_id_{mask_type}>'])
            if additional_tokens['additional_special_tokens']:
                tokenizer.add_special_tokens(additional_tokens)

        if tokenizer_cfg.library == 'sentencepiece':
            # NOTE: This is an ugly way to support both NeMo-Megatron trained checkpoints and huggingface checkpoints.
            # Huggingface and Google checkpoints will add sentinel tokens first (right after the base vocabulary), but in NeMo-Megatron, we add <cls>, <sep>, <mask>, <pad>, <bos> etc. beofore sentinel tokens <extra_id_xx>.
            if add_sentinel_tokens_first:
                if tokenizer_cfg.get('num_sentinel_tokens', 0) > 0:
                    cls._add_sentinel_tokens(
                        tokenizer, tokenizer_cfg.num_sentinel_tokens, add_sentinel_tokens_in_reverse_order
                    )
                cls._add_base_special_tokens(tokenizer, is_huggingface_converted_model=True)
            else:
                cls._add_base_special_tokens(tokenizer, is_huggingface_converted_model=False)
                if tokenizer_cfg.get('num_sentinel_tokens', 0) > 0:
                    cls._add_sentinel_tokens(
                        tokenizer, tokenizer_cfg.num_sentinel_tokens, add_sentinel_tokens_in_reverse_order
                    )

            if dataset_type == "ul2":
                for mask_type in ['r', 's', 'x']:
                    if len(tokenizer.text_to_ids(f'▁<extra_id_{mask_type}>')) == 1:
                        tokenizer.special_token_to_id[f'<extra_id_{mask_type}>'] = tokenizer.text_to_ids(
                            f'<extra_id_{mask_type}>'
                        )[0]
                    else:
                        tokenizer.add_special_tokens([f'<extra_id_{mask_type}>'])

    def build_train_valid_test_datasets(self):
        logging.info(f'Building {self.model_name} datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self._cfg.global_batch_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[
                1
            ] = 1  # This is to make sure we only have one epoch on every validation iteration

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self._cfg,
            trainer=self.trainer,
            tokenizer=self.tokenizer,
            data_prefix=self._cfg.data.data_prefix,
            data_impl=self._cfg.data.data_impl,
            splits_string=self._cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            max_seq_length=self._cfg.data.seq_length,
            masked_lm_prob=self._cfg.data.masked_lm_prob,
            short_seq_prob=self._cfg.data.short_seq_prob,
            seed=self._cfg.seed,
            skip_warmup=self._cfg.data.skip_warmup,
            dataset_type=self._cfg.data.get('dataset_type', self.model_name.lower()),
            max_ngram_size=self._cfg.data.get('max_ngram_size', 10),
            mean_ngram_size=self._cfg.data.get('mean_ngram_size', None),
            geometric_dist=self._cfg.data.get('geometric_dist', True),
            permutation=self._cfg.data.get('permutation', False),
            whole_word_masking=self._cfg.data.get('whole_word_masking', True),
            favor_long_ngrams=self._cfg.data.get('favor_long_ngrams', False),
            respect_document_boundaries=self._cfg.data.get('respect_document_boundaries', True),
            data_impl_kwargs=self._cfg.data.get('data_impl_kwargs', {}),
            # additional arguments from child classes
            **self._build_train_valid_test_datasets_kwargs,
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building {self.model_name} datasets.')

        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting in between a step
        self._reconfigure_limit_batches(self.trainer.limit_val_batches, self._validation_dl, 'val')

        return self._train_ds, self._validation_ds, self._test_ds

    def list_available_models(self):
        pass
