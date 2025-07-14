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
import hashlib
import json
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from random import sample
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union

import numpy as np
import torch
from datasets import DatasetDict, load_dataset

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.llm.gpt.data.retrieval import CustomRetrievalDataModule
from nemo.collections.llm.gpt.data.utils import _get_samples_mapping, _JSONLMemMapDataset
from nemo.core.classes import Dataset
from nemo.lightning.base import NEMO_DATASETS_CACHE

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


def get_dataset_root(name: str) -> Path:
    """Retrieve the root path for the dataset. Create the folder if not exists."""
    output = Path(NEMO_DATASETS_CACHE) / name
    output.mkdir(parents=True, exist_ok=True)

    return output


def create_reranker_dataset(
    path: Path,
    tokenizer: "TokenizerSpec",
    seq_length: int = 2048,
    add_bos: bool = False,
    add_eos: bool = True,
    seed: int = 1234,
    index_mapping_dir: Optional[str] = None,
    truncation_method: str = 'right',
    memmap_workers: int = 2,
    data_type: str = 'train',
    num_hard_negatives: int = 4,
    negative_sample_strategy: Literal["random", "first"] = 'first',
    question_key: str = 'query',
    pos_key: str = 'pos_doc',
    neg_key: str = 'neg_doc',
    **kwargs,
) -> "ReRankerDataset":
    """Create ReRankerDataset for reranking training."""

    return ReRankerDataset(
        file_path=str(path),
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        add_bos=add_bos,
        add_eos=add_eos,
        memmap_workers=memmap_workers,
        seed=seed,
        index_mapping_dir=index_mapping_dir,
        truncation_method=truncation_method,
        data_type=data_type,
        num_hard_negatives=num_hard_negatives,
        negative_sample_strategy=negative_sample_strategy,
        question_key=question_key,
        pos_key=pos_key,
        neg_key=neg_key,
        **kwargs,
    )


class ReRankerDataset(Dataset):
    """A dataset class for training reranking models that handles
    query-document pairs with positive and negative examples.

    This dataset processes JSONL files containing query-document triplets
    (query, positive document, negative documents) and prepares them for reranking model training.
    It supports various tokenization options, sequence length constraints,
    and negative sampling strategies.

    The dataset expects each example to contain:
    - A query/question
    - One or more positive documents (relevant to the query)
    - Multiple negative documents (irrelevant to the query)

    During processing, it:
    1. Formats each query-document pair with appropriate separators
    2. Applies tokenization with optional BOS/EOS tokens
    3. Handles sequence length constraints through truncation
    4. Samples negative examples based on the specified strategy
    5. Prepares attention masks and position IDs for model input

    The collate function combines multiple examples into batches, handling padding and attention masks
    appropriately for the reranking task.

    Args:
        file_path (str): Path to a JSONL dataset with (query,pos_doc,neg_doc) triplets.
        tokenizer (TokenizerSpec): Tokenizer for processing text.
        max_seq_length (int, optional): Maximum sequence length for each example. Defaults to 1024.
        min_seq_length (int, optional): Minimum sequence length for each example. Defaults to 1.
        add_bos (bool, optional): Whether to add beginning of sequence token. Defaults to True.
        add_eos (bool, optional): Whether to add end of sequence token. Defaults to True.
        max_num_samples (int, optional): Maximum number of samples to load. Defaults to None.
        seed (int, optional): Random seed for data shuffling. Defaults to 1234.
        index_mapping_dir (str, optional): Directory to save index mapping. Defaults to None.
        virtual_tokens (int, optional): Number of virtual tokens to add. Defaults to 0.
        memmap_workers (Optional[int], optional): Number of workers for memmap loading. Defaults to None.
        truncation_method (str, optional): Truncation method ('left' or 'right'). Defaults to 'right'.
        special_tokens (Optional[Mapping[str, str]], optional): Special tokens for formatting. Defaults to None.
        data_type (str, optional): Type of data ('train', 'query', or 'doc'). Defaults to 'train'.
        num_hard_negatives (int, optional): Number of negative examples to use. Defaults to 4.
        negative_sample_strategy (Literal["random", "first"], optional): Strategy for sampling negatives.
        Defaults to 'first'.
        question_key (str, optional): Key for question in input data. Defaults to 'question'.
        pos_key (str, optional): Key for positive document in input data. Defaults to 'pos_doc'.
        neg_key (str, optional): Key for negative documents in input data. Defaults to 'neg_doc'.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = True,
        add_eos: bool = True,
        max_num_samples: int = None,
        seed: int = 1234,
        index_mapping_dir: str = None,
        virtual_tokens: int = 0,
        memmap_workers: Optional[int] = None,
        truncation_method: str = 'right',
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        data_type: str = 'train',  # train, query or doc
        num_hard_negatives: int = 4,
        negative_sample_strategy: Literal["random", "first"] = 'first',
        question_key: str = 'question',
        pos_key: str = 'pos_doc',
        neg_key: str = 'neg_doc',
    ):
        """
        file_path: Path to a JSONL dataset with (query,pos_doc,neg_doc) triplets in jsonl format.
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec.
        max_seq_length (int): maximum sequence length for each dataset examples.
            Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset.
            Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load.
            This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        index_mapping_dir: Directory to save the index mapping to.
            If None, will write to the same folder as the dataset.
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}.
            Default: {
                        'system_turn_start': '<extra_id_0>',
                        'turn_start': '<extra_id_1>',
                        'label_start': '<extra_id_2>',
                        'end_of_turn': '\n',
                        'end_of_name": '\n'
                    }
        negative_sample_strategy: Strategy for negative samples. Options: ['random', 'first']
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.index_mapping_dir = index_mapping_dir
        self.virtual_tokens = virtual_tokens
        self.truncation_method = truncation_method
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id else self.tokenizer.eos_id
        self.negative_sample_strategy = negative_sample_strategy
        assert (
            truncation_method == 'left' or truncation_method == 'right'
        ), 'truncation_method must be either "left" or "right"'
        assert (
            negative_sample_strategy == 'random' or negative_sample_strategy == 'first'
        ), 'negative_sample_strategy must be either "random" or "first"'
        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens
        self.data_type = data_type
        self.num_hard_negatives = num_hard_negatives
        self.question_key = question_key
        self.pos_key = pos_key
        self.neg_key = neg_key

        # check if file_path is JSON file or JSONL file
        if file_path.endswith(".json"):
            # Convert JSON file to JSONL file
            logging.warning(f"Converting JSON file to JSONL file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
            with open(file_path.replace(".json", ".jsonl"), "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            file_path = file_path.replace(".json", ".jsonl")

        self.indexed_dataset = _JSONLMemMapDataset(
            dataset_paths=[file_path],
            tokenizer=None,
            header_lines=0,
            index_mapping_dir=index_mapping_dir,
            workers=memmap_workers,
        )
        # Will be None after this call if `max_num_samples` is None
        self.samples_mapping = None
        self._build_samples_mapping()
        logging.warn(
            f"Creating ReRankerDataset with seed={self.seed},\n"
            f"add_bos={self.add_bos}, add_eos={self.add_eos},\n"
            f"max_seq_length={self.max_seq_length}, min_seq_length={self.min_seq_length},\n"
            f"pad_token_id={self.pad_token_id}, negative_sample_strategy={self.negative_sample_strategy},\n"
            f"num_hard_negatives={self.num_hard_negatives}."
        )

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            self.samples_mapping = _get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=self.file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=self.file_path.split('/')[-1],
                binary_head=False,
                index_mapping_dir=self.index_mapping_dir,
            )
        else:
            self.samples_mapping = None

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.indexed_dataset)
        else:
            assert self.samples_mapping is not None
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):
                idx = idx.item()

        if idx is not None:
            assert idx < len(self.indexed_dataset)
        else:
            idx = -1
        # idx may < 0 because we pad_samples_to_global_batch_size, e.g. id = -1
        if idx < 0:
            idx = len(self) + idx
            auto_gen_idx = True
        else:
            auto_gen_idx = False
        try:
            example = self.indexed_dataset[idx]
            if auto_gen_idx:
                example['__AUTOGENERATED__'] = True
        except Exception as e:
            logging.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e
        return self._process_example(example)

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        metadata = {k: v for k, v in example.items()}

        question = example[self.question_key]
        pos_doc = example[self.pos_key]
        neg_doc = example[self.neg_key]

        # Only need one positive document for CE
        pos_doc = pos_doc[0] if isinstance(pos_doc, list) else pos_doc
        assert len(neg_doc) >= self.num_hard_negatives, "Error: not enough negative documents"
        if self.negative_sample_strategy == 'random':
            neg_doc = sample(neg_doc, k=self.num_hard_negatives)
        elif self.negative_sample_strategy == 'first':
            neg_doc = neg_doc[: self.num_hard_negatives]
        else:
            raise ValueError(f"Invalid negative sample strategy: {self.negative_sample_strategy}")
        assert len(neg_doc) == self.num_hard_negatives, "Error in sampling required number of hard negatives"

        # Construct 1 question + positive document, and self.num_hard_negatives question + negative documents
        def format_text(q, p):
            return f"question:{q} \n \n passage:{p}"

        positive = self.tokenizer.text_to_ids(format_text(question, pos_doc))
        negatives = [self.tokenizer.text_to_ids(format_text(question, ex)) for ex in neg_doc]

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens for ptuning (if used)
            positive = [self.tokenizer.eos_id] * self.virtual_tokens + positive  # type: ignore
            negatives = [[self.tokenizer.eos_id] * self.virtual_tokens + n for n in negatives]  # type: ignore

        if self.add_bos:
            positive = [self.tokenizer.bos_id] + positive  # type: ignore
            negatives = [[self.tokenizer.bos_id] + n for n in negatives]  # type: ignore

        # TODO: (@adithyare) should probably add a warning before truncation
        positive = positive[: self.max_seq_length - 1]
        negatives = [n[: self.max_seq_length - 1] for n in negatives]

        if self.add_eos:
            positive = positive + [self.tokenizer.eos_id]  # type: ignore
            negatives = [n + [self.tokenizer.eos_id] for n in negatives]  # type: ignore

        processed_example = {
            'positive': positive,
            'negatives': negatives,
            'metadata': metadata,
        }
        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _ceil_to_nearest(self, n, m):
        return (n + m - 1) // m * m

    def _collate_item(self, item, max_length):
        item = self._maybe_cast_to_list(item)
        pad_id = self.pad_token_id
        if self.truncation_method == 'left':
            item = [[pad_id] * (max_length - len(x)) + x for x in item]
        else:
            item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    @torch.no_grad()
    def _create_attention_mask2(self, max_length, item_length):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.zeros(max_length)
        if self.truncation_method == 'left':
            # input ids:      [pad] [pad] token token |
            # attention mask: 0      0    1     1
            attention_mask[max_length - item_length :] = 1
        else:
            # input ids:      token token [pad] [pad] |
            # attention mask: 1     1     0      0
            attention_mask[:item_length] = 1
        return attention_mask

    def collate_fn(self, batch):
        """
        Collate query passage together
        """
        input_ids = []
        metadata = []
        lengths = []
        max_length = -1

        # Flatten the batch
        # In the case of a micro batch size = 2, self.num_hard_negatives = 4,
        # we will have 2 * (1 + 4) = 10 examples in the batch
        # where the first 5 examples corresponds to the first question,
        # and the last 5 examples corresponds to the second question
        for item in batch:
            metadata.append(item['metadata'])
            input_ids.append(item['positive'])
            lengths.append(len(item['positive']))
            for nd in item['negatives']:
                input_ids.append(nd)
                lengths.append(len(nd))
            max_length = max(max_length, len(item['positive']), *(len(nd) for nd in item['negatives']))

        max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 16))
        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask2(max_length, len) for len in lengths]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(self._collate_item(input_ids, max_length=max_length))

        processed_batch = {
            'input_ids': input_ids,
            'token_type_ids': torch.zeros_like(input_ids),
            'attention_mask': attention_mask,
            'metadata': metadata,
            'position_ids': position_ids,
        }

        return processed_batch


class CustomReRankerDataModule(CustomRetrievalDataModule):
    """A data module for managing reranking datasets that handles data loading, preprocessing, and batching.

    This module extends CustomRetrievalDataModule to provide specialized functionality for reranking tasks.
    It manages the creation and organization of training, validation, and test datasets for reranking models,
    with support for automatic dataset splitting and various data loading configurations.

    The module can work with either:
    1. A single data file that will be automatically split into train/val/test sets
    2. Separate files for training, validation, and testing

    Key features:
    - Automatic dataset splitting with configurable ratios
    - Support for both JSON and JSONL file formats
    - Configurable batch sizes and data loading parameters
    - Efficient data loading with memory mapping
    - Support for packed sequence specifications
    - Customizable data keys for query and document fields

    Args:
        data_root (Union[str, List[str]]): Path(s) to the training data file(s) in JSON/JSONL format.
        val_root (Optional[str]): Path to validation data file. If None, will split from data_root.
        test_root (Optional[str]): Path to test data file. If None, will split from data_root.
        val_ratio (Optional[float]): Ratio of data to use for validation when splitting. Defaults to 0.04.
        test_ratio (Optional[float]): Ratio of data to use for testing when splitting. Defaults to 0.01.
        dataset_identifier (Optional[str]): Unique identifier for the dataset. If None, generated from data_root.
        seq_length (int): Maximum sequence length for model input. Defaults to 2048.
        tokenizer (Optional[TokenizerSpec]): Tokenizer for text processing. Defaults to None.
        micro_batch_size (int): Batch size for each training step. Defaults to 4.
        global_batch_size (int): Total batch size across all GPUs. Defaults to 8.
        rampup_batch_size (Optional[List[int]]): Batch sizes for training rampup. Defaults to None.
        force_redownload (bool): Whether to force redownload of dataset. Defaults to False.
        delete_raw (bool): Whether to delete raw data after processing. Defaults to True.
        seed (int): Random seed for reproducibility. Defaults to 1234.
        memmap_workers (int): Number of workers for memory-mapped file loading. Defaults to 1.
        num_workers (int): Number of workers for data loading. Defaults to 8.
        pin_memory (bool): Whether to pin memory for faster GPU transfer. Defaults to True.
        persistent_workers (bool): Whether to keep workers alive between epochs. Defaults to False.
        packed_sequence_specs (Optional[PackedSequenceSpecs]): Specifications for packed sequences. Defaults to None.
        query_key (str): Key for query field in data. Defaults to "question".
        pos_doc_key (str): Key for positive document field in data. Defaults to "pos_doc".
        neg_doc_key (str): Key for negative document field in data. Defaults to "neg_doc".
        dataset_kwargs (Optional[Dict[str, Any]]): Additional arguments for dataset creation. Defaults to None.
    """

    def __init__(
        self,
        data_root: Union[str, List[str]],
        val_root: Optional[str] = None,
        test_root: Optional[str] = None,
        val_ratio: Optional[float] = 0.04,
        test_ratio: Optional[float] = 0.01,
        dataset_identifier: Optional[str] = None,
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = True,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence_specs: Optional["PackedSequenceSpecs"] = None,
        query_key: str = "question",
        pos_doc_key: str = "pos_doc",
        neg_doc_key: str = "neg_doc",
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Custom DataModule for Finetuning reranking Dataset.

        Args:
            data_root (Union[str, List[str]]): The JSON/JSONL data file(s) used for training/validation/test.
                if val_root/test_root is not present, data_root will be split to training and val/test based on
                val_ratio/test_ratio.
            val_root (Optional[str]): The JSON/JSONL data file used for validation. If not provided, validation set
                will be split from data_root.
            test_root (Optional[str]): The JSON/JSONL data file used for test. If not provided, test set
                will be split from data_root.
            val_ratio (Optional[float]): The ratio of validation set when splitting from data_root.
            test_ratio (Optional[float]): The ratio of test set when splitting from data_root.
            dataset_identifier (str): Dataset identifier when saving the dataset to NEMO_HOME.
            seq_length (int, optional): The maximum sequence length for the input and output text. Defaults to 2048.
            tokenizer (Optional[TokenizerSpec], optional): The tokenizer to use for preprocessing the text.
                If not provided, a Megatron GPT2 BPE tokenizer will be used.
            micro_batch_size (int, optional): The micro batch size for training. Defaults to 4.
            global_batch_size (int, optional): The global batch size for training. Defaults to 8.
            rampup_batch_size (Optional[List[int]], optional): A list of batch sizes for ramping up during training.
                Defaults to None.
            seed (int, optional): The random seed for data shuffling. Defaults to 1234.
            memmap_workers (int, optional): The number of worker processes for loading data using TextMemMapDataset.
                Defaults to 1.
            num_workers (int, optional): The number of worker processes for data loading. Defaults to 8.
            pin_memory (bool, optional): Whether to pin memory during data loading for faster GPU training.
                Defaults to True.
            persistent_workers (bool, optional): Whether to keep data loading workers persistent across epochs.
                Defaults to False.
            dataset_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments to pass into the GPTSFTDataset class
        """
        if dataset_identifier is None:
            # Create dataset_identifier based on MD5 of data_root
            dataset_identifier = hashlib.md5(str(data_root).encode()).hexdigest()
        self.dataset_identifier = dataset_identifier
        super().__init__(
            data_root=data_root,
            val_root=val_root,
            test_root=test_root,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            dataset_identifier=dataset_identifier,
            seq_length=seq_length,
            tokenizer=tokenizer,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            force_redownload=force_redownload,
            delete_raw=delete_raw,
            seed=seed,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            packed_sequence_specs=packed_sequence_specs,
            query_key=query_key,
            pos_doc_key=pos_doc_key,
            neg_doc_key=neg_doc_key,
            dataset_kwargs=dataset_kwargs,
        )

    @lru_cache
    def _create_dataset(self, path, **kwargs):
        return create_reranker_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            **kwargs,
        )


class SpecterReRankerDataModule(CustomReRankerDataModule):
    """A data module for fine-tuning on the Specter dataset.

    This class inherits from the `CustomReRankerDataModule` class and is specifically designed for fine-tuning models
    on the SPECTER Datasets. It handles data download, preprocessing, splitting, and preparing the data
    in a format suitable for training, validation, and testing.

    Args:
        force_redownload (bool, optional): Whether to force re-download the dataset even if it exists locally.
                                           Defaults to False.
        delete_raw (bool, optional): Whether to delete the raw downloaded dataset after preprocessing.
                                     Defaults to True.
        See FineTuningDataModule for the other args
    """

    def __init__(
        self,
        dataset_root: str = None,
        seq_length: int = 512,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = True,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        dataset_kwargs: Optional[Dict[str, Any]] = {},
    ):
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw
        self.dataset_root = get_dataset_root("specter")
        self.seed = seed

        self.prepare_data()
        super().__init__(
            data_root=get_dataset_root("specter") / "training.jsonl",
            val_root=get_dataset_root("specter") / "validation.jsonl",
            test_root=get_dataset_root("specter") / "test.jsonl",
            seq_length=seq_length,
            tokenizer=tokenizer,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            seed=seed,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            dataset_kwargs={
                "num_hard_negatives": 1,
                **dataset_kwargs,
            },
        )

    def prepare_data(self) -> None:
        """Prepare dataset for fine-tuning."""
        # if train file is specified, no need to do anything
        if not self.train_path.exists() or self.force_redownload:
            dset = self._download_data()
            self._preprocess_and_split_data(dset)

    def _download_data(self):
        logging.info(f"Downloading {self.__class__.__name__}...")
        return load_dataset(
            "sentence-transformers/specter",
            "triplet",
            cache_dir=str(self.dataset_root),
            download_mode="force_redownload" if self.force_redownload else None,
        )

    def _preprocess_and_split_data(self, dset: DatasetDict, train_ratio: float = 0.80, val_ratio: float = 0.15):
        """Preprocesses and splits the downloaded dataset into training, validation, and test sets.

        Args:
            dset (DatasetDict): The downloaded dataset object.
            split_val_from_train (bool, optional): Whether to split the validation set from the training set.
                If False, the validation set is split from the test set. Defaults to True.
            val_proportion (float, optional): The proportion of the training or test set to be used
                for the validation split. Defaults to 0.05.
        """
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")

        test_ratio = 1 - train_ratio - val_ratio
        save_splits = {}
        dataset = dset.get('train')
        split_dataset = dataset.train_test_split(test_size=val_ratio + test_ratio, seed=self.seed)
        split_dataset2 = split_dataset['test'].train_test_split(
            test_size=test_ratio / (val_ratio + test_ratio), seed=self.seed
        )
        save_splits['training'] = split_dataset['train']
        save_splits['validation'] = split_dataset2['train']
        save_splits['test'] = split_dataset2['test']

        for split_name, dataset in save_splits.items():
            output_file = self.dataset_root / f"{split_name}.jsonl"
            with output_file.open("w", encoding="utf-8") as f:
                for o in dataset:
                    f.write(
                        json.dumps({"question": o["anchor"], "pos_doc": o["positive"], "neg_doc": [o["negative"]]})
                        + "\n"
                    )

            logging.info(f"{split_name} split saved to {output_file}")

        if self.delete_raw:
            for p in self.dataset_root.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                elif '.jsonl' not in str(p.name):
                    p.unlink()

    def reconfigure_limit_batches(self):
        """No need to reconfigure trainer.limit_val_batches for finetuning"""
        return

    @lru_cache
    def _create_dataset(self, path, **kwargs):
        return create_reranker_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            question_key="question",
            **kwargs,
        )
