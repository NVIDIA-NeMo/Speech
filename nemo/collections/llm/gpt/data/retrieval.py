# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os.path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datasets import Dataset

from nemo.collections.llm.bert.data.fine_tuning import FineTuningDataModule
from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


# Custom Retrieval Data Module loaded with json file
class CustomRetrievalDataModule(FineTuningDataModule):
    """ """

    def __init__(
        self,
        data_root: str,
        dataset_identifier: str = "custom_retrieval_dataset",
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
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw

        assert packed_sequence_specs is None, "RetrievalDataModule does not support packed sequences."
        assert os.path.exists(data_root), "Data root does not exist."
        self.query_key = query_key
        self.pos_doc_key = pos_doc_key
        self.neg_doc_key = neg_doc_key
        self.unprocessed_root = data_root
        super().__init__(
            dataset_root=get_dataset_root(dataset_identifier),
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
            dataset_kwargs=dataset_kwargs,
        )

    def prepare_data(self) -> None:
        """Prepare data if not split already."""
        if not self.train_path.exists() or self.force_redownload:
            self._preprocess_and_split_data()
        super().prepare_data()

    def _preprocess_and_split_data(self, train_ratio: float = 0.95, val_ratio: float = 0.04):
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")

        test_ratio = 1 - train_ratio - val_ratio
        save_splits = {}
        dataset = Dataset.from_list(json.load(open(self.unprocessed_root, 'r')))
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
                    # We only write one positive document for now
                    # All negative document are written
                    pos_doc = o[self.pos_doc_key][0] if isinstance(o[self.pos_doc_key], list) else o[self.pos_doc_key]
                    neg_doc = o[self.neg_doc_key] if isinstance(o[self.pos_doc_key], list) else [o[self.neg_doc_key]]
                    f.write(json.dumps({"query": o[self.query_key], "pos_doc": pos_doc, "neg_doc": neg_doc}) + "\n")

            logging.info(f"{split_name} split saved to {output_file}")
