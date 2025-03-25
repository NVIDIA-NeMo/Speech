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
from unittest.mock import patch

import torch
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon.config import ImageToken, LLaVATemplateConfig, MultiModalSampleConfig
from nemo.collections.vlm.llava_next.data.sample import (
    LlavaNextTextRawBatch,
    LlavaNextTextSample,
    PackedLlavaNextTextSample,
)
from nemo.collections.vlm.llava_next.data.task_encoder import LlavaNextTaskEncoder
from nemo.collections.vlm.llava_next.data.vqa_sample_encoder import LlavaNextSampleEncoder


class TestLlavaNextTaskEncoder(unittest.TestCase):

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
        self.encoder = LlavaNextTaskEncoder(
            tokenizer=self.tokenizer, image_processor=self.image_processor, multimodal_sample_config=self.config
        )

    def test_batch(self):
        # Import VQA sample types
        from megatron.energon import VQASample

        # Create VQA sample encoder
        vqa_encoder = LlavaNextSampleEncoder(
            tokenizer=self.tokenizer, image_processor=self.image_processor, multimodal_sample_config=self.config
        )

        # Create various VQA samples with different context/answer lengths and image sizes
        vqa_samples = [
            # Sample 1: Short context, short answer, small image
            VQASample(
                __key__="sample1",
                context="What is in this image?",
                answers="A cat.",
                image=torch.rand(3, 224, 224),  # Small image
                __restore_key__=None,
                __subflavor__=None,
                __subflavors__=[],
            ),
            # Sample 2: Medium context, medium answer, medium image
            VQASample(
                __key__="sample2",
                context="Can you describe what's happening in this image and identify the main objects?",
                answers="The image shows a living room with a sofa, coffee table, and two people sitting and talking.",
                image=torch.rand(3, 336, 336),  # Medium image
                __restore_key__=None,
                __subflavor__=None,
                __subflavors__=[],
            ),
            # Sample 3: Long context, long answer, large image
            VQASample(
                __key__="sample3",
                context="This is a complex scene with multiple elements. Can you analyze what's happening, describe the setting, identify any people or objects present?",
                answers="The image depicts an outdoor cafe scene on a busy street. There are approximately 8-10 people sitting at various tables, enjoying their meals and conversations. The setting appears to be in a European city, given the architecture visible in the background.",
                image=torch.rand(3, 448, 448),  # Large image
                __restore_key__=None,
                __subflavor__=None,
                __subflavors__=[],
            ),
        ]

        # Encode each sample
        encoded_samples = []
        for sample in vqa_samples:
            output_sample = LlavaNextTextSample()
            encoded_sample = vqa_encoder.encode(sample, output_sample)
            encoded_samples.append(encoded_sample)

        # Create batch using the encoder
        batch = self.encoder.batch(encoded_samples)

        # Print shapes of key tensors in batch
        print(f"Batch tokens shape: {batch.tokens.shape}")  # [num_samples, max_seq_len]
        print(f"Batch labels shape: {batch.labels.shape}")  # [num_samples, max_seq_len]
        print(f"Batch images shape: {batch.images.shape}")  # [total_tiles, channels, height, width]
        print(f"Batch num_media_tiles: {batch.num_media_tiles}")  # [num_samples]

        # Verify batch properties
        self.assertIsInstance(batch, LlavaNextTextRawBatch)
        self.assertEqual(len(batch.__keys__), len(encoded_samples))
        self.assertEqual(batch.tokens.shape[0], len(encoded_samples))
        self.assertEqual(batch.tokens.shape[1], max(sample.tokens.shape[0] for sample in encoded_samples))
        self.assertEqual(batch.labels.shape[0], len(encoded_samples))
        self.assertEqual(batch.labels.shape[1], max(sample.labels.shape[0] for sample in encoded_samples))
        self.assertEqual(batch.images.shape[0], sum(batch.num_media_tiles).item())
        self.assertEqual(len(batch.num_media_tiles), len(encoded_samples))

    @patch('nemo.collections.vlm.neva.data.sequence_packing.greedy_knapsack')
    @patch('nemo.collections.vlm.neva.data.sequence_packing.predict_seq_len_with_padding')
    def test_select_samples_to_pack(self, mock_predict_seq_len_with_padding, mock_greedy_knapsack):
        # Configure encoder with sequence packing enabled
        packing_encoder = LlavaNextTaskEncoder(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            multimodal_sample_config=self.config,
            packed_sequence=True,
            packed_sequence_size=1024,
        )

        # Create test samples with different token lengths and image sizes
        samples = [
            # Sample 1: Short sequence, small image
            LlavaNextTextSample(
                __key__="sample1",
                tokens=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                labels=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                images=torch.rand(1, 3, 224, 224),  # 1 tile, small image
                loss_mask=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                num_media_tiles=1,
                image_sizes=torch.tensor([[224, 224]], dtype=torch.long),
                attention_mask=torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            ),
            # Sample 2: Medium sequence, medium image
            LlavaNextTextSample(
                __key__="sample2",
                tokens=torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.long),
                labels=torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.long),
                images=torch.rand(2, 3, 336, 336),  # 2 tiles, medium image
                loss_mask=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                num_media_tiles=2,
                image_sizes=torch.tensor([[336, 336]], dtype=torch.long),
                attention_mask=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
            ),
            # Sample 3: Long sequence, large image
            LlavaNextTextSample(
                __key__="sample3",
                tokens=torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=torch.long),
                labels=torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=torch.long),
                images=torch.rand(4, 3, 448, 448),  # 4 tiles, large image
                loss_mask=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                num_media_tiles=4,
                image_sizes=torch.tensor([[448, 448]], dtype=torch.long),
                attention_mask=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
            ),
        ]

        # Configure mocks
        mock_predict_seq_len_with_padding.side_effect = [5, 9, 13]  # Return token lengths
        mock_greedy_knapsack.return_value = [[samples[0], samples[1]], [samples[2]]]  # Return packed groups

        # Call the method
        result = packing_encoder.select_samples_to_pack(samples)

        # Verify the results
        mock_greedy_knapsack.assert_called_once()
        self.assertEqual(len(result), 2)  # Two bins created
        self.assertEqual(len(result[0]), 2)  # Two samples in first bin
        self.assertEqual(len(result[1]), 1)  # One sample in second bin
        self.assertEqual(result[0][0].__key__, "sample1")
        self.assertEqual(result[0][1].__key__, "sample2")
        self.assertEqual(result[1][0].__key__, "sample3")

    @patch('nemo.collections.vlm.llava_next.data.utils.convert_to_packed_llava_next')
    def test_pack_selected_samples(self, mock_convert_to_packed):
        # Configure encoder with sequence packing enabled
        packing_encoder = LlavaNextTaskEncoder(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            multimodal_sample_config=self.config,
            packed_sequence=True,
            packed_sequence_size=1024,
        )

        # Create samples to pack with different token lengths and image sizes
        samples = [
            # Sample 1: Short sequence, small image
            LlavaNextTextSample(
                __key__="sample1",
                tokens=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                labels=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                images=torch.rand(1, 3, 336, 336),  # 1 tile, small image
                loss_mask=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                num_media_tiles=1,
                image_sizes=torch.tensor([[336, 336]], dtype=torch.long),
                attention_mask=torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            ),
            # Sample 2: Medium sequence, medium image
            LlavaNextTextSample(
                __key__="sample2",
                tokens=torch.tensor([6, 7, 8, 9, 10, 11, 12], dtype=torch.long),
                labels=torch.tensor([6, 7, 8, 9, 10, 11, 12], dtype=torch.long),
                images=torch.rand(2, 3, 336, 336),  # 2 tiles, medium image
                loss_mask=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                num_media_tiles=2,
                image_sizes=torch.tensor([[336, 336]], dtype=torch.long),
                attention_mask=torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
            ),
        ]

        # Configure mock for convert_to_packed_llava_next
        # Returned values need to match the combined lengths of the samples above
        mock_convert_to_packed.return_value = (
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),  # packed_tokens
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),  # packed_labels
            torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]),  # packed_position_ids
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # packed_loss_mask
            {
                "sample_lengths": [5, 7],
                "sequence_index_map": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
            },  # packed_seq_params
        )

        # Call the method
        result = packing_encoder.pack_selected_samples(samples)

        # Verify the result
        self.assertIsInstance(result, PackedLlavaNextTextSample)
        self.assertEqual(result.__key__, "sample1,sample2")
        self.assertEqual(result.images.shape[0], 3)  # 1+2 tiles

        # Update assertions to match actual tensor shapes
        # The tokens are padded to length 128 and batched with dim 1
        self.assertEqual(result.tokens.shape, torch.Size([1, 128]))  # Padded from original 12 tokens to 128
        self.assertEqual(result.labels.shape, torch.Size([1, 128]))  # Similarly padded
        self.assertEqual(result.loss_mask.shape, torch.Size([1, 128]))  # Similarly padded

        # Check for packed_seq_params fields
        # cu_seqlens_q/kv [0,5,12] represents cumulative sequence lengths:
        # - 0: Start index of first sequence
        # - 5: End of first sequence (length 5) / Start of second sequence
        # - 12: End of second sequence (length 7)
        self.assertIsNotNone(result.packed_seq_params)
        self.assertEqual(result.packed_seq_params.qkv_format, 'thd')
        self.assertEqual(len(result.packed_seq_params.cu_seqlens_q), 3)  # [0, 5, 12]
        self.assertEqual(result.packed_seq_params.cu_seqlens_q[0].item(), 0)  # Start index
        self.assertEqual(result.packed_seq_params.cu_seqlens_q[1].item(), 5)  # After first sequence
        self.assertEqual(result.packed_seq_params.cu_seqlens_q[2].item(), 12)  # After second sequence

        # Check num_media_tiles
        self.assertEqual(result.num_media_tiles.shape, torch.Size([2]))  # 2 images worth of tiles
        self.assertEqual(result.image_sizes.shape[0], 2)  # 2 images worth of sizes


if __name__ == '__main__':
    unittest.main()
