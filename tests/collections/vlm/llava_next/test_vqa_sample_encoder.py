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
from megatron.energon import VQASample
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon.config import ImageToken, LLaVATemplateConfig, MultiModalSampleConfig
from nemo.collections.vlm.llava_next.data.sample import LlavaNextTextSample
from nemo.collections.vlm.llava_next.data.vqa_sample_encoder import LlavaNextSampleEncoder


class TestLlavaNextSampleEncoder(unittest.TestCase):

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
            conversation_template_config=LLaVATemplateConfig(
                system="I am an AI assistant", roles=["user", "assistant"], stop_string=" </s>"
            ),
        )
        self.encoder = LlavaNextSampleEncoder(self.tokenizer, self.image_processor, self.config)

    def test_process_image(self):
        test_image = torch.rand(3, 224, 224)
        processed_image = self.encoder.process_image(test_image)
        self.assertIsInstance(processed_image, torch.Tensor)
        self.assertEqual(processed_image.dim(), 4)  # num_tiles, C, H, W format

    def test_encode(self):
        # Create dummy input sample
        input_sample = VQASample(
            __key__="test_sample",
            context="What is in this image?",
            answers="A test image.",
            image=torch.rand(3, 224, 224),
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
        )

        # Create output sample
        output_sample = LlavaNextTextSample()

        # Call encode
        result = self.encoder.encode(input_sample, output_sample)

        # Verify basic output properties
        self.assertEqual(result.__key__, "test_sample")
        self.assertIsNotNone(result.images)
        self.assertIsNotNone(result.tokens)
        self.assertIsNotNone(result.labels)
        self.assertIsNotNone(result.loss_mask)
        self.assertIsNotNone(result.attention_mask)
        self.assertIsNotNone(result.image_sizes)
        self.assertGreater(result.num_media_tiles, 0)


if __name__ == '__main__':
    unittest.main()
