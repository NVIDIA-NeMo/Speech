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

import unittest

from datasets import Dataset

from nemo.collections.llm.gpt.data import HFDatasetDataModulePacked
from nemo.collections.llm.gpt.data.hf_dataset_packed_sequence import HFDatasetPackedSequenceHelper


class TestPacking(unittest.TestCase):
    def setUp(self):
        # Mock dataset with sample sequences
        self.mock_data = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "loss_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6, 7, 8], "labels": [4, 5, 6, 7, 8], "loss_mask": [1, 1, 1, 1, 1]},
            {"input_ids": [9, 10, 11, 12], "labels": [9, 10, 11, 12], "loss_mask": [1, 1, 1, 1]},
        ]
        self.dataset = Dataset.from_list(self.mock_data)

    def test_basic_packing(self):
        """Test basic packing without splitting"""
        helper = HFDatasetPackedSequenceHelper(self.dataset, "train")
        packed = helper.pack(packed_sequence_size=8, split_across_pack=False, max_packs=2)

        # Verify pack counts
        self.assertEqual(len(packed), 2)

        # Verify first pack contents
        self.assertEqual(packed[0]["tokens"], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(packed[0]["seq_lens"], [3, 5])

    def test_split_across_pack(self):
        """Test splitting samples across packs"""
        helper = HFDatasetPackedSequenceHelper(self.dataset, "train")
        packed = helper.pack(packed_sequence_size=7, split_across_pack=True, max_packs=3)

        # Verify split samples
        self.assertEqual(packed[0]["tokens"][:7], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(packed[1]["tokens"], [8, 9, 10, 11, 12, 0, 0])

    def test_position_ids_padding(self):
        """Test position ID continuation during padding"""
        helper = HFDatasetPackedSequenceHelper(self.dataset.select([0]), "train")
        packed = helper.pack(packed_sequence_size=5, split_across_pack=False, max_packs=1)

        # Original positions: [0,1,2]
        # Padded positions should continue: [3,4]
        self.assertEqual(packed[0]["position_ids"], [0, 1, 2, 3, 4])

    def test_error_handling(self):
        """Test oversize sample without splitting"""
        helper = HFDatasetPackedSequenceHelper(self.dataset, "train")
        with self.assertRaises(ValueError):
            helper.pack(packed_sequence_size=2, split_across_pack=False, max_packs=None)

    def test_dataloader_integration(self):
        """Test full datamodule integration"""
        datamodule = HFDatasetDataModulePacked(
            path_or_dataset=self.dataset, packed_sequence_size=8, split_across_pack=False, split="train"
        )

        loader = datamodule.train_dataloader()
        batch = next(iter(loader))

        # Verify batch structure
        self.assertIn("tokens", batch)
        self.assertIn("position_ids", batch)
        self.assertIn("seq_lens", batch)
        self.assertEqual(batch["tokens"].shape[1], 8)


if __name__ == "__main__":
    unittest.main()
