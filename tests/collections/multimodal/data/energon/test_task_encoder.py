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
import torch
from megatron.energon import InterleavedSample, SimilarityInterleavedSample, VQASample
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon import (
    ImageTextSample,
    ImageToken,
    MultiModalSampleConfig,
    VQASampleEncoder,
)
from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder


class TestMultiModalTaskEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        cls.tokenizer = cls.processor.tokenizer
        cls.image_processor = cls.processor.image_processor

    def setUp(self):
        self.config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200), ignore_place_holder=-100
        )
        self.encoder = MultiModalTaskEncoder(
            tokenizer=self.tokenizer, image_processor=self.image_processor, multimodal_sample_config=self.config
        )

    def test_register_encoder(self):
        mock_encoder = VQASampleEncoder(self.tokenizer, self.image_processor, self.config)
        self.encoder.register_encoder("CustomSample", mock_encoder)
        self.assertIn("CustomSample", self.encoder.encoders)

    def test_encode_sample_vqa(self):
        sample = VQASample(
            __key__="sample_key",
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
            context="What is this?",
            answers="This is a test.",
            image=torch.rand(3, 224, 224),
        )
        encoded_sample = self.encoder.encode_sample(sample)
        self.assertIsInstance(encoded_sample, ImageTextSample)
        self.assertIsNotNone(encoded_sample.tokens)
        self.assertIsNotNone(encoded_sample.labels)

    def test_encode_sample_interleaved(self):
        sample = InterleavedSample(
            __key__="interleaved_sample",
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
            sequence=["This is a test.", torch.rand(3, 224, 224)],
        )
        encoded_sample = self.encoder.encode_sample(sample)
        self.assertIsInstance(encoded_sample, ImageTextSample)
        self.assertIsNotNone(encoded_sample.tokens)
        self.assertIsNotNone(encoded_sample.labels)

    def test_encode_sample_similarity_interleaved(self):
        sample = SimilarityInterleavedSample(
            __key__="sample_key",
            images=[torch.rand(3, 224, 224), torch.rand(3, 224, 224)],
            matched_text_indices=[0, 2],
            texts=["This is the first sentence.", "This is the second sentence.", "This is the third sentence."],
            similarity_matrix=None,
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
        )
        encoded_sample = self.encoder.encode_sample(sample)
        self.assertIsInstance(encoded_sample, ImageTextSample)
        self.assertIsNotNone(encoded_sample.tokens)
        self.assertIsNotNone(encoded_sample.labels)

    def test_batch(self):
        samples = [
            ImageTextSample(
                __key__="sample1",
                tokens=torch.tensor([1, 2, 3]),
                labels=torch.tensor([1, 2, 3]),
                images=torch.rand(1, 3, 224, 224),
                loss_mask=torch.tensor([1.0, 1.0, 1.0]),
            ),
            ImageTextSample(
                __key__="sample2",
                tokens=torch.tensor([1, 2, 3, 4, 5, 6]),
                labels=torch.tensor([1, 2, 3, 4, 5, 6]),
                images=torch.rand(1, 3, 224, 224),
                loss_mask=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            ),
        ]
        batch = self.encoder.batch(samples)
        self.assertIsInstance(batch, ImageTextRawBatch)
        self.assertEqual(batch.images.shape[0], len(samples))
        self.assertEqual(batch.tokens.shape[0], len(samples))
        self.assertEqual(batch.tokens.shape[1], max(sample.tokens.shape[0] for sample in samples))
        self.assertEqual(batch.labels.shape[0], len(samples))
        self.assertEqual(batch.labels.shape[1], max(sample.labels.shape[0] for sample in samples))
        self.assertEqual(batch.loss_mask.shape[0], len(samples))

    def test_encode_batch(self):
        samples = [
            ImageTextSample(
                __key__="sample1",
                tokens=torch.tensor([1, 2, 3]),
                labels=torch.tensor([1, 2, 3]),
                images=torch.rand(1, 3, 224, 224),
                loss_mask=torch.tensor([1.0, 1.0, 1.0]),
            ),
            ImageTextSample(
                __key__="sample2",
                tokens=torch.tensor([4, 5, 6]),
                labels=torch.tensor([4, 5, 6]),
                images=torch.rand(1, 3, 224, 224),
                loss_mask=torch.tensor([1.0, 1.0, 1.0]),
            ),
        ]
        batch = self.encoder.batch(samples)
        encoded_batch = self.encoder.encode_batch(batch)
        self.assertIsInstance(encoded_batch, dict)
        self.assertIn('position_ids', encoded_batch)
        self.assertIn('tokens', encoded_batch)
        self.assertIn('labels', encoded_batch)
        self.assertIn('loss_mask', encoded_batch)
        self.assertIn('attention_mask', encoded_batch)
        self.assertEqual(encoded_batch['tokens'].shape[0], len(samples))


if __name__ == '__main__':
    unittest.main()
