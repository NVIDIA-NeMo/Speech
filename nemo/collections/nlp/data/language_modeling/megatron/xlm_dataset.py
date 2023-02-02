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

"""XLM-Style datasets"""
from typing import Dict, List

import numpy as np

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.common.sequence_to_sequence_dataset import (
    BinarizedMemmapSequenceToSequenceDataset,
    TextMemmapSequenceToSequenceDataset,
)
from nemo.collections.nlp.data.language_modeling.megatron.bert_dataset import (
    build_training_sample as build_training_sample_bert,
)
from nemo.collections.nlp.data.language_modeling.megatron.length_distribution_type import LengthDistribution
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import T5Dataset
from nemo.collections.nlp.data.language_modeling.megatron.ul2_dataset import UL2Dataset


class CrossLingualBERTDataset(BinarizedMemmapSequenceToSequenceDataset):
    """Cross-lingual BERT dataset similar to the translation-language modeling objective in the XLM paper (https://arxiv.org/abs/1901.07291)"""

    def __init__(
        self,
        src_dataset_prefix: str,
        tgt_dataset_prefix: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        seed: int = 1234,
        max_num_samples: int = None,
        masked_lm_prob: float = 0.15,
    ):
        super().__init__(
            src_dataset_prefix=src_dataset_prefix,
            tgt_dataset_prefix=tgt_dataset_prefix,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length,
            max_tgt_seq_length=max_tgt_seq_length,
            seed=seed,
            max_num_samples=max_num_samples,
        )
        assert src_tokenizer == tgt_tokenizer
        # Vocab stuff.
        self.vocab = src_tokenizer.vocab
        self.vocab_id_list = list(range(self.src_tokenizer.original_vocab_size))
        self.vocab_id_to_token_dict = {idx: token for idx, token in enumerate(self.vocab)}
        self.cls_id = src_tokenizer.cls_id
        self.sep_id = src_tokenizer.sep_id
        self.mask_id = src_tokenizer.mask_id
        self.pad_id = src_tokenizer.pad_id
        self.bos_id = src_tokenizer.bos_id
        self.eos_id = src_tokenizer.eos_id
        self.max_seq_length = max_src_seq_length + max_tgt_seq_length
        self.masked_lm_prob = masked_lm_prob

    def __getitem__(self, idx):
        src, tgt = super()._get_sample(idx)
        if len(src) > self.max_src_seq_length:
            src = src[: self.max_src_seq_length]

        if len(tgt) > self.max_tgt_seq_length - 1:  # -1 here to account for the <sep> token that gets added.
            tgt = tgt[: self.max_tgt_seq_length]

        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2 ** 32))
        # Potentially swap src, tgt with a 50% chance to avoid learning associations based on position in the sequence.
        swap_src_tgt = np_rng.randint(0, 2)
        if swap_src_tgt == 0:
            sample = [np.concatenate((src, [self.sep_id], tgt))]
        elif swap_src_tgt == 1:
            sample = [np.concatenate((tgt, [self.sep_id], src))]

        return build_training_sample_bert(
            sample=sample,
            target_seq_length=sample[0].shape[0],
            max_seq_length=self.max_seq_length,  # needed for padding
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
            pad_id=self.pad_id,
            masked_lm_prob=self.masked_lm_prob,
            np_rng=np_rng,
            binary_head=False,
            whole_word_masking=False,
            skip_masking_id=self.sep_id,
        )

    # Skip the parent collate function, since we don't need it for this dataset.
    def collate_fn(self, batch):
        return batch


class CrossLingualMakedSequenceToSequenceDataset(BinarizedMemmapSequenceToSequenceDataset):
    def __init__(
        self,
        src_dataset_prefix: str,
        tgt_dataset_prefix: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        max_seq_length_dec: int,
        seed: int = 1234,
        max_num_samples: int = None,
        masked_lm_prob: float = 0.15,
        extreme_masked_lm_prob: float = 0.5,
        max_ngram_size: int = 10,
        mean_ngram_size: int = None,
        min_ngram_size: int = 1,
        extreme_max_ngram_size: int = 128,
        extreme_mean_ngram_size: int = 64,
        extreme_min_ngram_size: int = 32,
        extreme_ngram_span_length_distribution: LengthDistribution = LengthDistribution.truncated_normal,
        geometric_dist: bool = True,
        permutation: bool = False,
        favor_long_ngrams: bool = False,
        masking_type: str = "t5",
    ):
        super().__init__(
            src_dataset_prefix=src_dataset_prefix,
            tgt_dataset_prefix=tgt_dataset_prefix,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length,
            max_tgt_seq_length=max_tgt_seq_length,
            seed=seed,
            max_num_samples=max_num_samples,
        )
        self.max_seq_length_dec = max_seq_length_dec
        self.max_ngram_size = max_ngram_size
        self.mean_ngram_size = mean_ngram_size
        self.min_ngram_size = min_ngram_size
        self.geometric_dist = geometric_dist
        self.permutation = permutation
        self.favor_long_ngrams = favor_long_ngrams
        self.extreme_masked_lm_prob = extreme_masked_lm_prob
        self.extreme_max_ngram_size = extreme_max_ngram_size
        self.extreme_mean_ngram_size = extreme_mean_ngram_size
        self.extreme_min_ngram_size = extreme_min_ngram_size
        self.extreme_ngram_span_length_distribution = extreme_ngram_span_length_distribution
        self.masking_type = masking_type

        assert src_tokenizer == tgt_tokenizer
        # Vocab stuff.
        self.vocab_id_list = src_tokenizer.vocab
        self.vocab_id_to_token_dict = {idx: token for idx, token in enumerate(self.vocab_id_list)}
        self.cls_id = src_tokenizer.cls_id
        self.sep_id = src_tokenizer.sep_id
        self.mask_id = src_tokenizer.mask_id
        self.pad_id = src_tokenizer.pad_id
        self.bos_id = src_tokenizer.bos_id
        self.eos_id = src_tokenizer.eos_id
        self.max_seq_length = max_src_seq_length + max_tgt_seq_length
        self.masked_lm_prob = masked_lm_prob

        self.tokenizer_type = T5Dataset._determine_tokenizer_type(src_tokenizer, whole_word_masking=False)
        self._build()

    def _build(self):
        """
        Class-specific build method to be overridden by child classes.
        """
        self.sentinel_tokens = self.src_tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0

    def __getitem__(self, idx):
        src, tgt = super()._get_sample(idx)
        if len(src) > self.max_src_seq_length:
            src = src[: self.max_src_seq_length]

        if len(tgt) > self.max_tgt_seq_length - 1:  # -1 here to account for the <sep> token that gets added.
            tgt = tgt[: self.max_tgt_seq_length]

        np_rng = np.random.RandomState(seed=(self.seed + idx))

        return CrossLingualMakedSequenceToSequenceDataset.get_example(
            src=src,
            tgt=tgt,
            max_seq_length=self.max_seq_length,
            max_seq_length_dec=self.max_seq_length_dec,
            masked_lm_prob=self.masked_lm_prob,
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            sentinel_tokens=self.sentinel_tokens,
            max_ngram_size=self.max_ngram_size,
            np_rng=np_rng,
            mean_ngram_size=self.mean_ngram_size,
            min_ngram_size=self.min_ngram_size,
            extreme_masked_lm_prob=self.extreme_masked_lm_prob,
            extreme_max_ngram_size=self.extreme_max_ngram_size,
            extreme_mean_ngram_size=self.extreme_mean_ngram_size,
            extreme_min_ngram_size=self.extreme_min_ngram_size,
            extreme_ngram_span_length_distribution=self.extreme_ngram_span_length_distribution,
            favor_long_ngrams=self.favor_long_ngrams,
            permutation=self.permutation,
            geometric_dist=self.geometric_dist,
            tokenizer_type=self.tokenizer_type,
            masking_type=self.masking_type,
        )

    # Skip the parent collate function, since we don't need it for this dataset.
    def collate_fn(self, batch):
        return batch


class BinarizedMemmapCrossLingualMLMAndTranslationDataset(BinarizedMemmapSequenceToSequenceDataset):
    def __init__(
        self,
        src_dataset_prefix: str,
        tgt_dataset_prefix: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        src_language: str,
        tgt_language: str,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        max_seq_length_dec: int,
        seed: int = 1234,
        max_num_samples: int = None,
        masked_lm_prob: float = 0.15,
        extreme_masked_lm_prob: float = 0.5,
        max_ngram_size: int = 10,
        mean_ngram_size: int = None,
        min_ngram_size: int = 1,
        extreme_max_ngram_size: int = 128,
        extreme_mean_ngram_size: int = 64,
        extreme_min_ngram_size: int = 32,
        extreme_ngram_span_length_distribution: LengthDistribution = LengthDistribution.truncated_normal,
        prefix_lm_pivot_mean: float = 0.25,  # This is represented as a percentage of the total length.
        geometric_dist: bool = True,
        permutation: bool = False,
        favor_long_ngrams: bool = False,
        sampling_ratios: Dict[str, float] = {"x-masking": 0.25, "r-masking": 0.25, "s-masking": 0.25, "nmt": 0.25},
        sentinel_tokens: List[int] = None,
    ):
        super().__init__(
            src_dataset_prefix=src_dataset_prefix,
            tgt_dataset_prefix=tgt_dataset_prefix,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length
            - 1,  # -1 here to account for <sep> tokens and special prefix tokens to the encoder like <extra_id_x>, <extra_id_r>, etc.
            max_tgt_seq_length=max_tgt_seq_length
            - 1,  # -1 here to account for <sep> tokens and special prefix tokens to the encoder like <extra_id_x>, <extra_id_r>, etc.
            seed=seed,
            max_num_samples=max_num_samples,
            add_bos_to_enc=False,
            add_eos_to_enc=False,
        )
        self.max_seq_length_dec = max_seq_length_dec
        self.max_ngram_size = max_ngram_size
        self.mean_ngram_size = mean_ngram_size
        self.min_ngram_size = min_ngram_size
        self.geometric_dist = geometric_dist
        self.permutation = permutation
        self.favor_long_ngrams = favor_long_ngrams
        self.extreme_masked_lm_prob = extreme_masked_lm_prob
        self.extreme_max_ngram_size = extreme_max_ngram_size
        self.extreme_mean_ngram_size = extreme_mean_ngram_size
        self.extreme_min_ngram_size = extreme_min_ngram_size
        self.extreme_ngram_span_length_distribution = extreme_ngram_span_length_distribution
        self.prefix_lm_pivot_mean = prefix_lm_pivot_mean
        self.sampling_ratios = sampling_ratios
        self.src_language = src_language
        self.tgt_language = tgt_language

        # Vocab stuff.
        self.vocab_id_list = src_tokenizer.vocab
        self.vocab_id_to_token_dict = {idx: token for idx, token in enumerate(self.vocab_id_list)}
        self.cls_id = src_tokenizer.cls_id
        self.sep_id = src_tokenizer.sep_id
        self.mask_id = src_tokenizer.mask_id
        self.pad_id = src_tokenizer.pad_id
        self.bos_id = src_tokenizer.bos_id
        self.eos_id = src_tokenizer.eos_id
        self.max_seq_length = max_src_seq_length + max_tgt_seq_length - 1
        self.masked_lm_prob = masked_lm_prob
        self.sentinel_tokens = sentinel_tokens

        self.tokenizer_type = T5Dataset._determine_tokenizer_type(src_tokenizer, whole_word_masking=False)
        self._build()

    def _build(self):
        """
        Class-specific build method to be overridden by child classes.
        """
        if self.sentinel_tokens is None:
            self.sentinel_tokens = self.src_tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0

    @classmethod
    def create_sample(
        cls,
        np_rng: np.random.RandomState,
        src: List[int],
        tgt: List[int],
        sampling_ratios: Dict[str, float],
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        max_seq_length: int,
        max_seq_length_dec: int,
        src_language: str,
        tgt_language: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        sep_id: int,
        masked_lm_prob: float,
        extreme_masked_lm_prob: float,
        max_ngram_size: int,
        min_ngram_size: int,
        mean_ngram_size: int,
        extreme_max_ngram_size: int,
        extreme_min_ngram_size: int,
        extreme_mean_ngram_size: int,
        extreme_ngram_span_length_distribution: LengthDistribution,
        sentinel_tokens: List[int],
        prefix_lm_pivot_mean: float,
        vocab_id_list: List[int],
        vocab_id_to_token_dict: Dict[int, str],
        favor_long_ngrams: bool = False,
        permutation: bool = False,
        geometric_dist: bool = True,
        tokenizer_type: str = "wordpiece",
    ):
        # Determine which task to perform - NMT/T5/UL2 based on sampling ratios.
        task = np_rng.choice(list(sampling_ratios.keys()), p=list(sampling_ratios.values()))

        # Potentially swap src, tgt with a 50% chance to avoid learning associations based on position in the sequence.
        swap_src_tgt = np_rng.randint(0, 2)

        if len(src) > max_src_seq_length:
            src = src[:max_src_seq_length]

        # Tasks that are not NMT have a <sep> token so we need to account for this in the length, hence we truncate tgt to max_tgt_seq_length - 1.
        max_tgt_seq_length = max_tgt_seq_length - 1 if task != "nmt" else max_tgt_seq_length
        if len(tgt) > max_tgt_seq_length:
            tgt = tgt[:max_tgt_seq_length]

        if task == "nmt":
            # If src/tgt are swapped, also swap the prepend language token ID.
            if swap_src_tgt == 1:
                src, tgt = tgt, src
                prepend_id = f"<{src_language}>"
            else:
                prepend_id = f"<{tgt_language}>"

            text_dec = np.concatenate([[tgt_tokenizer.bos_id], tgt])
            labels = np.concatenate([tgt, [tgt_tokenizer.eos_id]])
            nmt_sample = {'text_enc': src, 'text_dec': text_dec, 'labels': labels}
            return UL2Dataset._prepend_mask_type_token(src_tokenizer, nmt_sample, prepend_id)

        if swap_src_tgt == 0:
            sample = [np.concatenate((src, [sep_id], tgt))]
        elif swap_src_tgt == 1:
            sample = [np.concatenate((tgt, [sep_id], src))]

        if task == "x-masking":
            return UL2Dataset.get_x_masking_training_sample(
                sample=sample,
                tokenizer=src_tokenizer,
                np_rng=np_rng,
                target_seq_length=sample[0].shape[0],
                max_seq_length=max_seq_length,  # -1 to account for the <extra_id_x> token that gets added after the sample is created.
                max_seq_length_dec=max_seq_length_dec,
                masked_lm_prob=masked_lm_prob,
                extreme_masked_lm_prob=extreme_masked_lm_prob,
                max_ngram_size=max_ngram_size,
                min_ngram_size=min_ngram_size,
                mean_ngram_size=mean_ngram_size,
                extreme_max_ngram_size=extreme_max_ngram_size,
                extreme_min_ngram_size=extreme_min_ngram_size,
                extreme_mean_ngram_size=extreme_mean_ngram_size,
                extreme_ngram_span_length_distribution=extreme_ngram_span_length_distribution,
                sentinel_tokens=sentinel_tokens,
                skip_masking_id=sep_id,
            )
        elif task == "s-masking":
            return UL2Dataset.get_s_masking_training_sample(
                sample=sample,
                np_rng=np_rng,
                max_seq_length_encoder=max_seq_length,  # -1 to account for the <extra_id_s> token that gets added after the sample is created.
                max_seq_length_decoder=max_seq_length_dec,
                tokenizer=src_tokenizer,
                prefix_lm_pivot_mean=prefix_lm_pivot_mean,
                pivot_distribution=extreme_ngram_span_length_distribution,
                add_eos=True,  # Most sentences are < max length in cross-lingual data, so we add an EOS to indicate to the model to stop.
            )
        elif task == "r-masking":
            return UL2Dataset.get_r_masking_training_sample(
                sample=sample,
                tokenizer=src_tokenizer,
                np_rng=np_rng,
                target_seq_length=sample[0].shape[0],
                max_seq_length=max_seq_length,  # -1 to account for the <extra_id_r> token that gets added after the sample is created.
                max_seq_length_dec=max_seq_length_dec,
                masked_lm_prob=masked_lm_prob,
                vocab_id_list=vocab_id_list,
                vocab_id_to_token_dict=vocab_id_to_token_dict,
                max_ngram_size=max_ngram_size,
                mean_ngram_size=mean_ngram_size,
                whole_word_masking=False,
                favor_long_ngrams=favor_long_ngrams,
                permutation=permutation,
                geometric_dist=geometric_dist,
                tokenizer_type=tokenizer_type,
                sentinel_tokens=sentinel_tokens,
                skip_masking_id=sep_id,
            )

    def __getitem__(self, idx):
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        src, tgt = super()._get_sample(idx)
        return BinarizedMemmapCrossLingualMLMAndTranslationDataset.create_sample(
            np_rng=np_rng,
            src=src,
            tgt=tgt,
            sampling_ratios=self.sampling_ratios,
            max_src_seq_length=self.max_src_seq_length,
            max_tgt_seq_length=self.max_tgt_seq_length,
            max_seq_length=self.max_seq_length,
            max_seq_length_dec=self.max_seq_length_dec,
            src_language=self.src_language,
            tgt_language=self.tgt_language,
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer,
            sep_id=self.sep_id,
            masked_lm_prob=self.masked_lm_prob,
            extreme_masked_lm_prob=self.extreme_masked_lm_prob,
            max_ngram_size=self.max_ngram_size,
            min_ngram_size=self.min_ngram_size,
            mean_ngram_size=self.mean_ngram_size,
            extreme_max_ngram_size=self.extreme_max_ngram_size,
            extreme_min_ngram_size=self.extreme_min_ngram_size,
            extreme_mean_ngram_size=self.extreme_mean_ngram_size,
            extreme_ngram_span_length_distribution=self.extreme_ngram_span_length_distribution,
            sentinel_tokens=self.sentinel_tokens,
            prefix_lm_pivot_mean=self.prefix_lm_pivot_mean,
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            favor_long_ngrams=self.favor_long_ngrams,
            permutation=self.permutation,
            geometric_dist=self.geometric_dist,
            tokenizer_type=self.tokenizer_type,
        )

    # NOTE: We want the parent's collate_fn to be used here since NMT examples are not padded even though the other task are.


class TextMemmapCrossLingualMLMAndTranslationDataset(TextMemmapSequenceToSequenceDataset):
    def __init__(
        self,
        src_file_name: str,
        tgt_file_name: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        src_language: str,
        tgt_language: str,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        max_seq_length_dec: int,
        seed: int = 1234,
        max_num_samples: int = None,
        masked_lm_prob: float = 0.15,
        extreme_masked_lm_prob: float = 0.5,
        max_ngram_size: int = 10,
        mean_ngram_size: int = None,
        min_ngram_size: int = 1,
        extreme_max_ngram_size: int = 128,
        extreme_mean_ngram_size: int = 64,
        extreme_min_ngram_size: int = 32,
        extreme_ngram_span_length_distribution: LengthDistribution = LengthDistribution.truncated_normal,
        prefix_lm_pivot_mean: float = 0.25,  # This is represented as a percentage of the total length.
        geometric_dist: bool = True,
        permutation: bool = False,
        favor_long_ngrams: bool = False,
        sampling_ratios: Dict[str, float] = {"x-masking": 0.25, "r-masking": 0.25, "s-masking": 0.25, "nmt": 0.25},
        sentinel_tokens: List[int] = None,
    ):
        super().__init__(
            src_file_name=src_file_name,
            tgt_file_name=tgt_file_name,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length
            - 1,  # -1 here to account for <sep> tokens and special prefix tokens to the encoder like <extra_id_x>, <extra_id_r>, etc.
            max_tgt_seq_length=max_tgt_seq_length
            - 1,  # -1 here to account for <sep> tokens and special prefix tokens to the encoder like <extra_id_x>, <extra_id_r>, etc.
            seed=seed,
            max_num_samples=max_num_samples,
            add_bos_to_enc=False,
            add_eos_to_enc=False,
        )
        self.max_seq_length_dec = max_seq_length_dec
        self.max_ngram_size = max_ngram_size
        self.mean_ngram_size = mean_ngram_size
        self.min_ngram_size = min_ngram_size
        self.geometric_dist = geometric_dist
        self.permutation = permutation
        self.favor_long_ngrams = favor_long_ngrams
        self.extreme_masked_lm_prob = extreme_masked_lm_prob
        self.extreme_max_ngram_size = extreme_max_ngram_size
        self.extreme_mean_ngram_size = extreme_mean_ngram_size
        self.extreme_min_ngram_size = extreme_min_ngram_size
        self.extreme_ngram_span_length_distribution = extreme_ngram_span_length_distribution
        self.prefix_lm_pivot_mean = prefix_lm_pivot_mean
        self.sampling_ratios = sampling_ratios
        self.src_language = src_language
        self.tgt_language = tgt_language

        # Vocab stuff.
        self.vocab_id_list = src_tokenizer.vocab
        self.vocab_id_to_token_dict = {idx: token for idx, token in enumerate(self.vocab_id_list)}
        self.cls_id = src_tokenizer.cls_id
        self.sep_id = src_tokenizer.sep_id
        self.mask_id = src_tokenizer.mask_id
        self.pad_id = src_tokenizer.pad_id
        self.bos_id = src_tokenizer.bos_id
        self.eos_id = src_tokenizer.eos_id
        self.max_seq_length = max_src_seq_length + max_tgt_seq_length - 1
        self.masked_lm_prob = masked_lm_prob
        self.sentinel_tokens = sentinel_tokens

        self.tokenizer_type = T5Dataset._determine_tokenizer_type(src_tokenizer, whole_word_masking=False)
        self._build()

    def _build(self):
        """
        Class-specific build method to be overridden by child classes.
        """
        if self.sentinel_tokens is None:
            self.sentinel_tokens = self.src_tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0

    def __getitem__(self, idx):
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        src, tgt = super()._get_sample(idx)
        return BinarizedMemmapCrossLingualMLMAndTranslationDataset.create_sample(
            np_rng=np_rng,
            src=src,
            tgt=tgt,
            sampling_ratios=self.sampling_ratios,
            max_src_seq_length=self.max_src_seq_length,
            max_tgt_seq_length=self.max_tgt_seq_length,
            max_seq_length=self.max_seq_length,
            max_seq_length_dec=self.max_seq_length_dec,
            src_language=self.src_language,
            tgt_language=self.tgt_language,
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer,
            sep_id=self.sep_id,
            masked_lm_prob=self.masked_lm_prob,
            extreme_masked_lm_prob=self.extreme_masked_lm_prob,
            max_ngram_size=self.max_ngram_size,
            min_ngram_size=self.min_ngram_size,
            mean_ngram_size=self.mean_ngram_size,
            extreme_max_ngram_size=self.extreme_max_ngram_size,
            extreme_min_ngram_size=self.extreme_min_ngram_size,
            extreme_mean_ngram_size=self.extreme_mean_ngram_size,
            extreme_ngram_span_length_distribution=self.extreme_ngram_span_length_distribution,
            sentinel_tokens=self.sentinel_tokens,
            prefix_lm_pivot_mean=self.prefix_lm_pivot_mean,
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            favor_long_ngrams=self.favor_long_ngrams,
            permutation=self.permutation,
            geometric_dist=self.geometric_dist,
            tokenizer_type=self.tokenizer_type,
        )

    # NOTE: We want the parent's collate_fn to be used here since NMT examples are not padded even though the other task are.
