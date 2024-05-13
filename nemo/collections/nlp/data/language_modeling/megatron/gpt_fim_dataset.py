# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Tuple

import numpy as np

from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.utils import logging

try:
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from megatron.core.datasets.utils import Split

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError) as e:

    GPTDataset = GPTDatasetConfig = IndexedDataset = Split = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False
    IMPORT_ERROR = e


class GPTFIMDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core GPT FIM datasets

    Attributes:
        fim: fill in the middle parameters config
    """

    def __init__(self, fim, **kwargs):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(IMPORT_ERROR)

        super().__init__(**kwargs)
        self.fim = fim


class GPTFIMDataset(GPTDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the
        MegatronDataset

        indexed_indices (np.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (GPTFIMDatasetConfig): The GPT-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: str,
        indexed_indices: np.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTFIMDatasetConfig,
    ) -> None:
        if not HAVE_MEGATRON_CORE:
            raise ImportError(IMPORT_ERROR)

        super().__init__(indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config)

        self.indexed_dataset = indexed_dataset
        self.np_rng = np.random.RandomState(seed=self.config.random_seed)
        logging.info(f"Initialized FIM RNG with seed = {self.config.random_seed}")
        # get FIM params
        self.fim_rate = self.config.fim.get('rate', 0.5)
        self.fim_spm_rate = self.config.fim.get('spm_rate', 0.5)
        self.fragment_fim_rate = self.config.fim.get('fragment_rate', 0.5)
        split_sample = self.config.fim.get('split_sample', None)
        self.fim_split_sample = self.config.tokenizer.tokens_to_ids(split_sample) if split_sample else None
        self.no_fim_prefix = self.config.fim.get('no_prefix', None)

        # get extra tokens ids
        fim_tokens = self.config.fim.extra_tokens
        fim_tokens = [fim_tokens.prefix, fim_tokens.middle, fim_tokens.suffix, fim_tokens.pad, fim_tokens.eod]
        fim_tokens_ids = self.config.tokenizer.tokens_to_ids(fim_tokens)
        (
            self.prefix_tok_id,
            self.middle_tok_id,
            self.suffix_tok_id,
            self.pad_tok_id,
            self.eod_tok_id,
        ) = fim_tokens_ids

    def _query_document_sample_shuffle_indices(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[np.ndarray, np.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.indexed_dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                sample_parts.append(self.indexed_dataset.get(self.document_index[i], offset=offset, length=length))

        sample = np.concatenate(sample_parts)

        sample_len = sample.shape[0]
        segment_breaks = np.argwhere(sample == self.eod_tok_id)
        np_rng = self.np_rng

        if segment_breaks.shape != (0, 1):  # then there is an EOD token in this example
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(segment_breaks):
                # Only permute non-empty segments.
                if loc - curr_start_position > 0:
                    # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                    permuted = self._fim_split_and_permute_sequence(sample[curr_start_position:loc], np_rng)
                    new_samples += [permuted, [self.eod_tok_id]]

                curr_start_position = loc + 1  # jump over the EOD token
            # Permute the segment after the last EOD
            permuted = self._fim_split_and_permute_sequence(sample[curr_start_position:], np_rng)
            new_samples.append(permuted)

            sample = np.concatenate(new_samples)
        else:
            sample = self._fim_split_and_permute_sequence(sample, np_rng)

        diff = sample.shape[0] - sample_len
        if diff > 0:  # too long
            sample = sample[:sample_len]
        elif diff < 0:  # too short
            sample = np.concatenate([sample, np.full((-1 * diff), self.pad_tok_id)])

        assert sample.shape[0] == sample_len

        return (
            np.array(sample, dtype=np.int64),
            np.array(document_ids, dtype=np.int64),
        )

    def _fim_permute_sequence(self, sequence, np_rng, rate):
        return self._permute(
            sequence,
            np_rng,
            rate,
            self.fim_spm_rate,
            self.config.tokenizer,
            truncate_or_pad=False,
            suffix_tok_id=self.suffix_tok_id,
            prefix_tok_id=self.prefix_tok_id,
            middle_tok_id=self.middle_tok_id,
            pad_tok_id=self.pad_tok_id,
            no_fim_prefix=self.no_fim_prefix,
        )

    def _fim_split_and_permute_sequence(self, sequence, np_rng):
        """
        If self.fim_split_sample is not None, split the sequence.
        Then apply FIM on the fragments, or the whole sequence if self.fim_split_sample is None.
        """
        if self.fim_split_sample is None:
            return self._fim_permute_sequence(sequence, np_rng, self.fim_rate)
        # fim_split_sample is set: split the sample on this token and permute each fragment separately.
        # Typically, if each sample is a repository, then we split again on the file level.
        # Each fragment is a file, and we permute the files.
        fragment_breaks = np.argwhere(sequence == self.fim_split_sample)
        if fragment_breaks.shape == (0, 1):
            # no split token in this sample
            return self._fim_permute_sequence(sequence, np_rng, self.fim_rate)
        if not np_rng.binomial(1, self.fim_rate):
            # don't do FIM preproc
            return sequence
        # Do FIM on each fragment
        curr_start_position = 0
        new_samples = []
        for loc in np.nditer(fragment_breaks):
            if loc - curr_start_position > 0:
                permuted = self._fim_permute_sequence(
                    sequence[curr_start_position:loc], np_rng, self.fragment_fim_rate
                )
                new_samples += [permuted, [self.fim_split_sample]]
            curr_start_position = loc + 1  # Jump over the split token
        # Permute the segment after the last split token
        permuted = self._fim_permute_sequence(sequence[curr_start_position:], np_rng, self.fragment_fim_rate)
        new_samples.append(permuted)

        return np.concatenate(new_samples)

    def _permute(
        self,
        sample,
        np_rng,
        fim_rate,
        fim_spm_rate,
        tokenizer,
        truncate_or_pad=True,
        suffix_tok_id=None,
        prefix_tok_id=None,
        middle_tok_id=None,
        pad_tok_id=None,
        no_fim_prefix=None,
    ):
        """
        Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
        Maintain the same sample length (if transform creates a few extra tokens, drop them).
        """
        if np_rng.binomial(1, fim_rate):  # sample bernoulli dist

            contents = tokenizer.ids_to_text(sample)

            # Do not apply FIM if the sample starts with no_fim_prefix
            if no_fim_prefix is not None and contents.startswith(no_fim_prefix):
                return sample

            try:
                # A boundary can be =0 (prefix will be empty)
                # a boundary can be =len(contents) (suffix will be empty)
                # The two boundaries can be equal (middle will be empty)
                boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
                boundaries.sort()
            except ValueError as e:
                print(len(contents), contents)
                print(e)
                raise e

            prefix = contents[: boundaries[0]]
            middle = contents[boundaries[0] : boundaries[1]]
            suffix = contents[boundaries[1] :]

            prefix = np.array([*tokenizer.text_to_ids(prefix)], dtype=np.int64)
            middle = np.array([*tokenizer.text_to_ids(middle)], dtype=np.int64)
            suffix = np.array([*tokenizer.text_to_ids(suffix)], dtype=np.int64)

            # here we truncate each given segment to fit the same length as it was before
            # A consequence is that we never reach the end of a file?
            # we should rather truncate at the context-level
            if truncate_or_pad:
                # need to make same length as the input. Take the 3 sentinel tokens into account
                new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
                diff = new_length - sample.shape[0]
                if diff > 0:  # too long
                    if (
                        suffix.shape[0] <= diff
                    ):  # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                        return sample, np_rng
                    suffix = suffix[: suffix.shape[0] - diff]
                elif diff < 0:  # too short
                    suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

            if np_rng.binomial(1, fim_spm_rate):
                # SPM (variant 2 from FIM paper)
                new_sample = np.concatenate([[prefix_tok_id, suffix_tok_id], suffix, [middle_tok_id], prefix, middle])
            else:
                # PSM
                new_sample = np.concatenate(
                    [[prefix_tok_id], prefix, [suffix_tok_id], suffix, [middle_tok_id], middle]
                )

        else:
            # don't do FIM preproc
            new_sample = sample

        return new_sample
