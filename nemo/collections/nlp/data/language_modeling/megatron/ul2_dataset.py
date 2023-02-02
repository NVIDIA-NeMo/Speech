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

"""UL2 Style dataset from https://arxiv.org/abs/2205.05131"""
import numpy as np

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import create_extreme_masked_lm_predictions
from nemo.collections.nlp.data.language_modeling.megatron.length_distribution_type import LengthDistribution
from nemo.collections.nlp.data.language_modeling.megatron.lm_adapted_t5_dataset import T5LMAdaptedDataset
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import T5Dataset


class UL2Dataset(T5Dataset):
    """ UL2 Dataset from https://arxiv.org/abs/2205.05131.
    Consists of three different objectives:
    1. Short span masking with small probabilities (ex: T5). Typically max ngram size of 5 with 0.15 mask prob.
    2. Extreme span masking with either large probabilities or large ngram sizes or both.
    3. Prefx-LM as in the T5 or LM-adapted T5 (prompt-tuning paper).
    """

    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        indexed_dataset,
        data_prefix,
        num_epochs,
        max_num_samples,
        max_seq_length,
        max_seq_length_dec,
        seed,
        masked_lm_prob=0.15,
        extreme_masked_lm_prob=0.5,
        short_seq_prob=0.0,
        min_ngram_size=2,
        max_ngram_size=10,
        mean_ngram_size=3,
        extreme_max_ngram_size=128,
        extreme_min_ngram_size=32,
        extreme_mean_ngram_size=64,
        prefix_lm_pivot_mean=0.25,  # This is represented as a percentage of the total length.
        ngram_span_length_distribution=LengthDistribution.geometric,
        extreme_ngram_span_length_distribution=LengthDistribution.truncated_normal,
        permutation=False,
        whole_word_masking=True,
        favor_long_ngrams=False,
        respect_document_boundaries=True,
        documents=None,
    ):
        super().__init__(
            cfg=cfg,
            trainer=trainer,
            tokenizer=tokenizer,
            name=name,
            indexed_dataset=indexed_dataset,
            data_prefix=data_prefix,
            num_epochs=num_epochs,
            max_num_samples=max_num_samples,
            max_seq_length=max_seq_length - 1,  # -1 to account for the added mask type token
            max_seq_length_dec=max_seq_length_dec,
            seed=seed,
            masked_lm_prob=masked_lm_prob,
            short_seq_prob=short_seq_prob,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=None,  # TODO: Determin if we want to actually pass mean ngram as an override to max here.
            geometric_dist=ngram_span_length_distribution == LengthDistribution.geometric,
            permutation=permutation,
            whole_word_masking=whole_word_masking,
            favor_long_ngrams=favor_long_ngrams,
            respect_document_boundaries=respect_document_boundaries,
            documents=documents,
        )
        self.mean_ngram_size = mean_ngram_size
        self.min_ngram_size = min_ngram_size
        self.extreme_masked_lm_prob = extreme_masked_lm_prob
        self.extreme_min_ngram_size = extreme_min_ngram_size
        self.extreme_max_ngram_size = extreme_max_ngram_size
        self.extreme_mean_ngram_size = extreme_mean_ngram_size
        self.ngram_span_length_distribution = ngram_span_length_distribution
        self.extreme_ngram_span_length_distribution = extreme_ngram_span_length_distribution
        self.prefix_lm_pivot_mean = prefix_lm_pivot_mean

    @classmethod
    def get_r_masking_training_sample(
        cls,
        sample,
        tokenizer,
        np_rng,
        target_seq_length: int,
        max_seq_length: int,
        max_seq_length_dec: int,
        masked_lm_prob: float,
        vocab_id_list: list,
        vocab_id_to_token_dict: dict,
        max_ngram_size: int,
        mean_ngram_size: int,
        whole_word_masking: bool,
        favor_long_ngrams: bool,
        permutation: bool,
        geometric_dist: bool,
        tokenizer_type: str,
        sentinel_tokens: list,
        skip_masking_id: int,
    ):
        # Call T5's build training sample for regular short span masking.
        sample = T5Dataset.build_training_sample(
            sample=sample,
            target_seq_length=target_seq_length,
            np_rng=np_rng,
            max_seq_length=max_seq_length,
            max_seq_length_dec=max_seq_length_dec,
            masked_lm_prob=masked_lm_prob,
            vocab_id_list=vocab_id_list,
            vocab_id_to_token_dict=vocab_id_to_token_dict,
            cls_id=tokenizer.cls_id,
            sep_id=tokenizer.sep_id,
            mask_id=tokenizer.mask_id,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=mean_ngram_size,
            whole_word_masking=whole_word_masking,
            favor_long_ngrams=favor_long_ngrams,
            permutation=permutation,
            geometric_dist=geometric_dist,
            tokenizer_type=tokenizer_type,
            sentinel_tokens=sentinel_tokens,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            skip_masking_id=skip_masking_id,
        )
        sample = UL2Dataset._prepend_mask_type_token(tokenizer, sample, '<extra_id_r>')
        return sample

    @classmethod
    def get_s_masking_training_sample(
        cls,
        sample,
        np_rng,
        max_seq_length_encoder: int,
        max_seq_length_decoder: int,
        tokenizer: TokenizerSpec,
        prefix_lm_pivot_mean: float,
        pivot_distribution: LengthDistribution,
        add_eos: bool = False,
    ):
        sample = [token for sentence in sample for token in sentence]
        sample = T5LMAdaptedDataset.get_prefix_lm_sample(
            sample=sample,
            max_seq_length_encoder=max_seq_length_encoder,
            max_seq_length_decoder=max_seq_length_decoder,  # We don't use max_seq_length_decoder here since we typically want to use long decoder sequences for better LM performance and we can do +1 because we don't need to add the UL2 token here.
            np_rng=np_rng,
            tokenizer=tokenizer,
            pivot_mean=prefix_lm_pivot_mean,
            pivot_distribution=pivot_distribution,
            add_eos=add_eos,
        )
        sample = UL2Dataset._prepend_mask_type_token(tokenizer, sample, '<extra_id_s>')
        return sample

    @classmethod
    def get_x_masking_training_sample(
        cls,
        sample,
        tokenizer,
        np_rng,
        target_seq_length: int,
        max_seq_length: int,
        max_seq_length_dec: int,
        masked_lm_prob: float,
        extreme_masked_lm_prob: float,
        max_ngram_size: int,
        min_ngram_size: int,
        mean_ngram_size: int,
        extreme_max_ngram_size: int,
        extreme_min_ngram_size: int,
        extreme_mean_ngram_size: int,
        extreme_ngram_span_length_distribution: LengthDistribution,
        sentinel_tokens: list,
        skip_masking_id: int,
    ):
        sample = UL2Dataset.build_extreme_masking_training_sample(
            sample=sample,
            target_seq_length=target_seq_length,
            np_rng=np_rng,
            max_seq_length=max_seq_length,
            max_seq_length_dec=max_seq_length_dec,
            masked_lm_prob=masked_lm_prob,
            extreme_masked_lm_prob=extreme_masked_lm_prob,
            mask_id=tokenizer.mask_id,
            max_ngram_size=max_ngram_size,
            min_ngram_size=min_ngram_size,
            extreme_max_ngram_size=extreme_max_ngram_size,
            extreme_mean_ngram_size=extreme_mean_ngram_size,
            extreme_min_ngram_size=extreme_min_ngram_size,
            extreme_ngram_span_length_distribution=extreme_ngram_span_length_distribution,
            mean_ngram_size=mean_ngram_size,
            sentinel_tokens=sentinel_tokens,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            skip_masking_id=skip_masking_id,
        )
        sample = UL2Dataset._prepend_mask_type_token(tokenizer, sample, '<extra_id_x>')
        return sample

    def __getitem__(self, idx):
        sample, seq_length = self._get_sample(idx)
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        masking_type = np_rng.randint(0, 3)  # 0: short span masking, 1: extreme masking, 2: prefix-LM
        if masking_type == 0:
            # Call T5's build training sample for regular short span masking.
            return UL2Dataset.get_r_masking_training_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                np_rng=np_rng,
                target_seq_length=seq_length,
                max_seq_length=self.max_seq_length,
                max_seq_length_dec=self.max_seq_length_dec,
                masked_lm_prob=self.masked_lm_prob,
                vocab_id_list=self.vocab_id_list,
                vocab_id_to_token_dict=self.vocab_id_to_token_dict,
                max_ngram_size=self.max_ngram_size,
                mean_ngram_size=self.mean_ngram_size,
                whole_word_masking=self.whole_word_masking,
                favor_long_ngrams=self.favor_long_ngrams,
                permutation=self.permutation,
                geometric_dist=self.geometric_dist,
                tokenizer_type=self.tokenizer_type,
                sentinel_tokens=self.sentinel_tokens,
                skip_masking_id=None,
            )
        elif masking_type == 1:
            return UL2Dataset.get_x_masking_training_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                np_rng=np_rng,
                target_seq_length=seq_length,
                max_seq_length=self.max_seq_length,
                max_seq_length_dec=self.max_seq_length_dec,
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
                skip_masking_id=None,
            )
        elif masking_type == 2:
            return UL2Dataset.get_s_masking_training_sample(
                sample=sample,
                np_rng=np_rng,
                max_seq_length_encoder=self.max_seq_length,
                max_seq_length_decoder=self.max_seq_length_dec,
                tokenizer=self.tokenizer,
                prefix_lm_pivot_mean=self.prefix_lm_pivot_mean,
                pivot_distribution=self.extreme_ngram_span_length_distribution,
            )

    @classmethod
    def _prepend_mask_type_token(cls, tokenizer, sample, token):
        token_id = tokenizer.text_to_ids(token)
        assert len(token_id) == 1, token
        token_id = token_id[0]
        text_enc = np.concatenate([[token_id], sample['text_enc']])
        sample['text_enc'] = text_enc
        if 'enc_mask' in sample:
            sample['enc_mask'] = np.concatenate([[1], sample['enc_mask']])
        return sample

    @classmethod
    def build_extreme_masking_training_sample(
        cls,
        sample,
        target_seq_length,
        np_rng,
        max_seq_length,
        max_seq_length_dec,
        masked_lm_prob,
        extreme_masked_lm_prob,
        mask_id,
        max_ngram_size,
        min_ngram_size,
        mean_ngram_size,
        extreme_max_ngram_size,
        extreme_mean_ngram_size,
        extreme_min_ngram_size,
        extreme_ngram_span_length_distribution,
        sentinel_tokens,
        bos_id,
        eos_id,
        pad_id,
        skip_masking_id=None,
    ):
        """Build training sample.
        Arguments:
            sample: A list of sentences in which each sentence is a list token ids.
            target_seq_length: Desired sequence length.
            max_seq_length: Maximum length of the sequence. All values are padded to
                this length.
            vocab_id_list: List of vocabulary ids. Used to pick a random id.
            vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
            cls_id: Start of example id.
            sep_id: Separator id.
            mask_id: Mask token id.
            pad_id: Padding token id.
            masked_lm_prob: Probability to mask tokens.
            np_rng: Random number genenrator. Note that this rng state should be
                  numpy and not python since python randint is inclusive for
                  the opper bound whereas the numpy one is exclusive.
            bos_id: start of decoder example id
            eos_id: end of generation id
            sentinel_tokens: unique value to be substituted for every replaced span
            tokenizer_type: wordpiece (BERT-style) or sentencepiece tokenizer. Used for whole word masking logic.
            max_ngram_size: maximum size of ngrams to be masked.
            mean_ngram_size: mean size of ngrams to be masked (only used if geometric_dist=True).
            geometric_dist: Uses a geometric distribution to sample ngram size.
            permutation: Permutes the ngrams.
            whole_word_masking: Always masks entire words instead of individual sub-word tokens.
            favor_long_ngrams: Favor longer ngrams over shorter ones.
            skip_masking_id: id of the token to that will never be masked.
        """
        assert target_seq_length <= max_seq_length

        # flatten sentences into one list
        tokens = [token for sentence in sample for token in sentence]

        # Truncate to `target_sequence_length`.
        max_num_tokens = target_seq_length
        tokens = tokens[:max_num_tokens]

        # Determine if we have a lot of masking or little masking. There are three cases:
        # 1. Small masking prob, large spans.
        # 2. Large masking prob, small spans.
        # 3. Large masking prob, large spans.
        task_type = np_rng.randint(0, 3)
        if task_type == 0:
            # Large spans, small masking prob
            max_ngram_size, mean_ngram_size, min_ngram_size, masked_lm_prob = (
                extreme_max_ngram_size,
                extreme_mean_ngram_size,
                extreme_min_ngram_size,
                masked_lm_prob,
            )
        elif task_type == 1:
            # Small spans, large masking prob
            max_ngram_size, mean_ngram_size, min_ngram_size, masked_lm_prob = (
                max_ngram_size,
                mean_ngram_size,
                min_ngram_size,
                extreme_masked_lm_prob,
            )
        else:
            # Large spans, large masking prob
            max_ngram_size, mean_ngram_size, min_ngram_size, masked_lm_prob = (
                extreme_max_ngram_size,
                extreme_mean_ngram_size,
                extreme_mean_ngram_size,
                extreme_masked_lm_prob,
            )

        # Masking.
        max_predictions_per_seq = masked_lm_prob * max_num_tokens

        lm_pred = create_extreme_masked_lm_predictions(
            tokens=tokens,
            masked_lm_prob=masked_lm_prob,
            mask_id=mask_id,
            max_predictions_per_seq=max_predictions_per_seq,
            np_rng=np_rng,
            max_ngram_size=max_ngram_size,
            min_ngram_size=min_ngram_size,
            mean_ngram_size=mean_ngram_size,
            span_length_distribution=extreme_ngram_span_length_distribution,
            skip_masking_id=skip_masking_id,
        )

        if masked_lm_prob == 0:
            (output_tokens, masked_positions, masked_labels) = lm_pred
            masked_spans = None
        else:
            (output_tokens, masked_positions, masked_labels, masked_spans) = lm_pred

        # Padding.
        tokens_enc, tokens_dec_in, labels, enc_mask, dec_mask, loss_mask = T5Dataset.pad_and_convert_to_numpy(
            output_tokens=output_tokens,
            masked_positions=masked_positions,
            masked_labels=masked_labels,
            masked_spans=masked_spans,
            sentinel_tokens=sentinel_tokens,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_seq_length=max_seq_length,
            max_seq_length_dec=max_seq_length_dec,
        )

        train_sample = {
            'text_enc': tokens_enc,
            'text_dec': tokens_dec_in,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
        }

        return train_sample
