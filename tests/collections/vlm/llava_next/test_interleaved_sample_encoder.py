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

import unittest

import torch
from megatron.energon import SimilarityInterleavedSample
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon.config import ImageToken, MultiModalSampleConfig
from nemo.collections.vlm.llava_next.data.interleaved_sample_encoder import LlavaNextSimilarityInterleavedSampleEncoder
from nemo.collections.vlm.llava_next.data.sample import LlavaNextTextSample


class TestLlavaNextSimilarityInterleavedSampleEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use actual processor
        cls.processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        cls.tokenizer = cls.processor.tokenizer
        cls.image_processor = cls.processor.image_processor

    def setUp(self):
        self.config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200),
            ignore_place_holder=-100,
            image_following_text=True,
        )
        self.encoder = LlavaNextSimilarityInterleavedSampleEncoder(self.tokenizer, self.image_processor, self.config)

    def test_process_image(self):
        test_image = torch.rand(3, 224, 224)
        processed_image = self.encoder.process_image(test_image)
        self.assertIsInstance(processed_image, torch.Tensor)
        self.assertEqual(processed_image.dim(), 4)  # num_tiles, C, H, W format

    def test_encode_image_following_text(self):
        # Create dummy input sample with image following text
        images = [torch.rand(3, 224, 224), torch.rand(3, 224, 224)]
        texts = ["This is the first text.", "This is the second text.", "This is the third text."]
        matched_text_indices = [0, 2]  # Images should be placed after texts at indices 0 and 2

        input_sample = SimilarityInterleavedSample(
            __key__="test_interleaved",
            images=images,
            texts=texts,
            similarity_matrix=None,  # Not used in encoding
            matched_text_indices=matched_text_indices,
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
        )

        # Create output sample
        output_sample = LlavaNextTextSample()

        # Call encode
        result = self.encoder.encode(input_sample, output_sample)

        # Verify basic output properties
        self.assertEqual(result.__key__, "test_interleaved")
        self.assertIsNotNone(result.images)
        self.assertIsNotNone(result.tokens)
        self.assertIsNotNone(result.labels)
        self.assertIsNotNone(result.loss_mask)
        self.assertIsNotNone(result.attention_mask)
        self.assertIsNotNone(result.image_sizes)
        self.assertGreater(len(result.num_media_tiles), 0)

        # The image sizes should match the number of images
        self.assertEqual(result.image_sizes.shape[0], len(images))

    def test_encode_image_before_text(self):
        # Set up encoder with image before text
        self.config.image_following_text = False
        self.encoder = LlavaNextSimilarityInterleavedSampleEncoder(self.tokenizer, self.image_processor, self.config)

        # Create dummy input sample with image before text
        images = [torch.rand(3, 224, 224)]
        texts = ["This is a sample text."]
        matched_text_indices = [0]  # Image should be placed before text at index 0

        input_sample = SimilarityInterleavedSample(
            __key__="test_before",
            images=images,
            texts=texts,
            similarity_matrix=None,
            matched_text_indices=matched_text_indices,
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
        )

        # Create output sample
        output_sample = LlavaNextTextSample()

        # Call encode
        result = self.encoder.encode(input_sample, output_sample)

        # Verify basic output properties
        self.assertEqual(result.__key__, "test_before")
        self.assertIsNotNone(result.images)
        self.assertIsNotNone(result.tokens)
        self.assertIsNotNone(result.labels)
        self.assertIsNotNone(result.loss_mask)
        self.assertIsNotNone(result.image_sizes)


if __name__ == '__main__':
    unittest.main()
