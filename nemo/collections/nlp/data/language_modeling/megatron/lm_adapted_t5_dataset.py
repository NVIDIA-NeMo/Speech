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
import numpy as np

from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset
from nemo.collections.nlp.data.language_modeling.megatron.length_distribution_type import LengthDistribution


class T5LMAdaptedDataset(GPTDataset):
    """
    Dataset for unlearning span corruption (https://arxiv.org/abs/2104.08691) in T5 models.
    Corresponds to the prefix-LM objective in the T5 paper (Table 3 in https://arxiv.org/abs/1910.10683).
    """

    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seed,
        max_seq_length_encoder,
        max_seq_length_decoder,
        **kwargs,
    ):
        self.max_seq_length_encoder = max_seq_length_encoder
        self.max_seq_length_decoder = max_seq_length_decoder
        self.seed = seed
        self.tokenizer = tokenizer
        super().__init__(
            cfg,
            trainer,
            tokenizer,
            name,
            data_prefix,
            documents,
            indexed_dataset,
            num_samples,
            self.max_seq_length_encoder
            + self.max_seq_length_decoder
            + 1,  # +1 because the decoder sequence gets truncated by one due to shifting to for teacher-forcing.
            seed,
        )

    @classmethod
    def get_prefix_lm_sample(
        cls,
        sample,
        max_seq_length_encoder,
        max_seq_length_decoder,
        np_rng,
        tokenizer,
        pivot_mean=0.25,
        pivot_distribution=LengthDistribution.uniform,
        add_eos=False,
    ):
        # get random split index
        if pivot_distribution == LengthDistribution.truncated_normal and (pivot_mean < 0.0 or pivot_mean > 1.0):
            raise ValueError(
                f"Invalid pivot_mean: {pivot_mean}. Must be in [0.0, 1.0]. It is a fraction of the encoder sequence length."
            )

        # 1) If the sample is larger than max encoder sequence length, use max encoder sequence length
        # 2) Otherwwise use sample length - 1 so that there is at least one token on the decoder.
        max_split_idx = min(len(sample) - 1, max_seq_length_encoder)

        if pivot_distribution == LengthDistribution.uniform:
            split_idx = np_rng.randint(0, max_split_idx)
        elif pivot_distribution == LengthDistribution.truncated_normal:
            loc = pivot_mean * max_split_idx
            split_idx = np.clip(int(np_rng.normal(loc=loc, scale=loc)), 0, max_split_idx,)
        else:
            raise ValueError(f"Invalid pivot_distribution: {pivot_distribution}")

        # Encoder inputs get truncated based on the split indx
        tokens_enc = np.concatenate(
            [sample[:split_idx], [tokenizer.pad_id] * (max_seq_length_encoder - split_idx)]
        ).astype(np.int64)

        # The decoder sequence is never truncated and is always of max decoder length.
        offset = 1 if add_eos else 0
        tokens_dec = sample[split_idx : split_idx + max_seq_length_decoder - offset]

        # NOTE: Add bos only and not eos because the model will always generate till max seq length.
        example = np.concatenate([[tokenizer.bos_id], tokens_dec])
        if add_eos:
            example = np.concatenate([example, [tokenizer.eos_id]])

        # Example can be + 1 over sequence length at this point since we'll be shifting by 1 to create the inputs and outputs to the decoder.
        assert len(example) <= max_seq_length_decoder + 1
        tokens_dec = np.concatenate(
            [example, [tokenizer.pad_id] * (max_seq_length_decoder - len(example) + 1)]
        ).astype(np.int64)

        # Shift sequences for teacher forcing
        tokens_dec_in = tokens_dec[:-1]
        labels = tokens_dec[1:]

        # Create attention masks
        enc_mask = (tokens_enc != tokenizer.pad_id).astype(np.int64)
        dec_mask = (tokens_dec_in != tokenizer.pad_id).astype(np.int64)

        loss_mask = dec_mask

        train_sample = {
            'text_enc': tokens_enc,
            'text_dec': tokens_dec_in,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
        }
        return train_sample

    def __getitem__(self, idx):
        text = super()._get_text(idx)
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        sample = T5LMAdaptedDataset.get_prefix_lm_sample(
            sample=text,
            max_seq_length_encoder=self.max_seq_length_encoder,
            max_seq_length_decoder=self.max_seq_length_decoder,
            np_rng=np_rng,
            tokenizer=self.tokenizer,
            pivot_distribution=LengthDistribution.uniform,
        )
        return sample
