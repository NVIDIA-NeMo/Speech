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

"""BART Style dataset."""

import numpy as np

from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import (
    create_masked_lm_predictions,
    get_samples_mapping,
)
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import T5Dataset


class BARTDataset(T5Dataset):
    # account for added tokens
    MAX_SEQ_LENGTH_DELTA = 2

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
        seed,
        masked_lm_prob=0.15,
        short_seq_prob=0.1,
        max_ngram_size=10,
        mean_ngram_size=None,
        geometric_dist=True,
        permutation=False,
        whole_word_masking=True,
        favor_long_ngrams=False,
        delete_mask_prob=0,
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
            max_seq_length=max_seq_length,
            max_seq_length_dec=None,
            seed=seed,
            masked_lm_prob=masked_lm_prob,
            short_seq_prob=short_seq_prob,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=mean_ngram_size,
            geometric_dist=geometric_dist,
            permutation=permutation,
            whole_word_masking=whole_word_masking,
            favor_long_ngrams=favor_long_ngrams,
            respect_document_boundaries=respect_document_boundaries,
            documents=documents,
        )

        # Params to store.
        self.delete_mask_prob = delete_mask_prob

    def _build(self):
        """
        Class-specific build method to be overridden by child classes.
        """
        pass

    def __getitem__(self, idx):
        np_rng = np.random.RandomState(seed=(self.seed + idx))

        sample, seq_length = self._get_sample(idx)

        # flatten sentences into one list
        tokens = [token for sentence in sample for token in sentence]

        # Truncate to `target_sequence_length`.
        max_num_tokens = seq_length
        tokens = tokens[:max_num_tokens]

        # Masking.
        max_predictions_per_seq = self.masked_lm_prob * max_num_tokens

        lm_pred = create_masked_lm_predictions(
            tokens=tokens,
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            masked_lm_prob=self.masked_lm_prob,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
            max_predictions_per_seq=max_predictions_per_seq,
            np_rng=np_rng,
            max_ngram_size=self.max_ngram_size,
            whole_word_masking=self.whole_word_masking,
            favor_long_ngrams=self.favor_long_ngrams,
            mean_ngram_size=self.mean_ngram_size,
            permutation=self.permutation,
            geometric_dist=self.geometric_dist,
            masking_style="t5",
            tokenizer_type=self.tokenizer_type,
        )

        if self.masked_lm_prob == 0:
            (output_tokens, masked_positions, masked_labels, _) = lm_pred
            masked_spans = None
        else:
            (output_tokens, masked_positions, masked_labels, _, masked_spans) = lm_pred

        # Padding.
        tokens_enc, tokens_dec_in, labels, enc_mask, dec_mask, loss_mask = self.pad_and_convert_to_numpy(
            tokens=tokens,
            output_tokens=output_tokens,
            masked_positions=masked_positions,
            masked_labels=masked_labels,
            masked_spans=masked_spans,
            np_rng=np_rng,
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

    def pad_and_convert_to_numpy(
        self, tokens, output_tokens, masked_positions, masked_labels, masked_spans=None, np_rng=None,
    ):
        """Pad sequences and convert them to numpy."""
        bart_decoder_in = [self.bos_id] + tokens
        bart_decoder_out = tokens + [self.eos_id]

        if masked_spans is not None:
            # construct bart input by collapsing multiple <mask> into one, and delete randomly
            bart_input = []
            (start_index, end_index) = (0, None)
            for span in masked_spans:
                end_index = span.index[0]
                bart_input.extend(output_tokens[start_index:end_index])
                # delete mask with probability delete_mask_prob
                if np_rng.rand() >= self.delete_mask_prob:
                    bart_input.append(self.mask_id)

                # the next start index is the token after the last span token
                start_index = span.index[-1] + 1

            # Add the remaining tokens to the BART input
            bart_input.extend(output_tokens[start_index:])
        else:
            bart_input = output_tokens

        # Some checks.
        # Encoder-side padding mask.
        num_tokens = len(bart_input)
        padding_length = self.max_seq_length - num_tokens
        assert padding_length >= 0
        assert len(masked_positions) == len(masked_labels)

        # Tokens..
        filler = [self.pad_id] * padding_length
        tokens_enc = np.array(bart_input + filler, dtype=np.int64)

        # Decoder-side padding mask.
        num_tokens_dec = len(bart_decoder_in)
        padding_length_dec = self.max_seq_length - num_tokens_dec
        assert padding_length_dec >= 0
        filler_dec = [self.pad_id] * padding_length_dec
        tokens_dec_in = np.array(bart_decoder_in + filler_dec, dtype=np.int64)

        # Create attention masks
        enc_mask = (tokens_enc != self.pad_id).astype(np.int64)
        dec_mask = (tokens_dec_in != self.pad_id).astype(np.int64)

        # Labels mask.
        labels = bart_decoder_out + ([-1] * padding_length_dec)
        labels = np.array(labels, dtype=np.int64)

        # Loss mask
        loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
        loss_mask = np.array(loss_mask, dtype=np.int64)

        return tokens_enc, tokens_dec_in, labels, enc_mask, dec_mask, loss_mask
