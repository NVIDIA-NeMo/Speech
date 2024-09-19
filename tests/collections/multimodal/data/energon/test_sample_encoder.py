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

import re
import unittest

import torch
from megatron.energon import InterleavedSample, SimilarityInterleavedSample, VQASample
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon import (
    BaseSampleEncoder,
    ImageTextSample,
    ImageToken,
    InterleavedSampleEncoder,
    LLaVATemplateConfig,
    MultiModalSampleConfig,
    SimilarityInterleavedEncoder,
    VQASampleEncoder,
)


class TestBaseSampleEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use the actual processor
        cls.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        cls.tokenizer = cls.processor.tokenizer
        cls.image_processor = cls.processor.image_processor
        cls.ignore_place_holder = -100

    def setUp(self):
        self.config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200), ignore_place_holder=self.ignore_place_holder
        )
        self.encoder = BaseSampleEncoder(self.tokenizer, self.image_processor, self.config)

    def test_process_image(self):
        image = torch.rand(3, 224, 524)  # Mock a random image tensor

        processed_image = self.encoder.process_image(image)

        self.assertEqual(processed_image.shape, (1, 1, 3, 336, 336))

    def test_compute_loss_mask(self):
        labels = torch.tensor([1, self.ignore_place_holder, 2, self.ignore_place_holder])
        expected_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss_mask = self.encoder.compute_loss_mask(labels)
        self.assertTrue(torch.equal(loss_mask, expected_mask))


class TestVQASampleEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        cls.tokenizer = cls.processor.tokenizer
        cls.image_processor = cls.processor.image_processor

    def setUp(self):
        self.config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200),
            ignore_place_holder=-100,
            conversation_template_config=LLaVATemplateConfig(
                system='Im groot', roles=['user', 'assistant'], stop_string=' </s>'
            ),
        )
        self.encoder = VQASampleEncoder(self.tokenizer, self.image_processor, self.config)

    def test_apply_prompt_template(self):
        sample = VQASample(
            __key__="sample_key",
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
            context="What is this?",
            answers="This is a test.",
            image=torch.rand(3, 224, 224),  # Provide a mock image tensor
        )

        prompt = self.encoder.apply_prompt_template(sample)

        system_message = re.escape(self.config.conversation_template_config.system)
        user_context = re.escape(sample.context)
        assistant_answer = re.escape(sample.answers)
        stop_string = "</s>"
        self.assertRegex(
            prompt,
            rf"{system_message}",
            msg="System message block does not match the expected format.",
        )

        self.assertTrue(prompt.endswith("</s>"), msg="Prompt does not end with the expected '</s>' sequence.")

        expected_pattern = rf"{system_message.strip()} USER: {user_context.strip()} ASSISTANT: {assistant_answer.strip()}{stop_string}"
        # Check that the entire prompt matches the expected pattern
        self.assertRegex(
            prompt, expected_pattern, msg=f"The entire prompt does not match the expected pattern. Got: {prompt}"
        )

    def test_tokenize_with_special_token(self):
        prompt = "This is a <image> test."
        tokens = self.encoder.tokenize(prompt)
        expected_tokens = self.tokenizer("This is a ", add_special_tokens=False).input_ids
        expected_tokens.append(self.config.image_token.token_id)
        expected_tokens.extend(self.tokenizer(" test.", add_special_tokens=False).input_ids)
        self.assertTrue(
            torch.equal(tokens, torch.tensor(expected_tokens, dtype=torch.long)),
            msg=f"Expected tokens {expected_tokens}, but got {tokens.tolist()}",
        )

    def test_tokenize_with_only_special_token(self):
        prompt = "<image>"
        tokens = self.encoder.tokenize(prompt)
        expected_tokens = []
        expected_tokens.append(self.config.image_token.token_id)
        self.assertTrue(
            torch.equal(tokens, torch.tensor(expected_tokens, dtype=torch.long)),
            msg=f"Expected tokens {expected_tokens}, but got {tokens.tolist()}",
        )

    def test_compute_labels(self):
        prompt = "Question: What is this. This is a test. </s>"
        answer_text = "This is a test."
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0]
        answer_tokens = self.tokenizer.encode(
            answer_text + self.config.conversation_template_config.stop_string,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        answer_start, answer_end = -1, -1
        for i in range(len(tokens) - len(answer_tokens) + 1):
            if torch.equal(tokens[i : i + len(answer_tokens)], answer_tokens):
                answer_start, answer_end = i, i + len(answer_tokens)
                break

        sample = VQASample(
            __key__="sample_key",
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
            context="Question: What is this?",
            answers="This is a test.",
            image=torch.rand(3, 224, 224),
        )

        labels = self.encoder.compute_labels(tokens, sample)
        expected_labels = torch.ones_like(tokens) * self.config.ignore_place_holder
        if answer_start != -1:
            expected_labels[answer_start:answer_end] = tokens[answer_start:answer_end]
        self.assertTrue(
            torch.equal(labels, expected_labels),
            msg=f"Expected labels {expected_labels.tolist()}, but got {labels.tolist()}",
        )

    def test_encode(self):
        input_sample = VQASample(
            __key__="sample_key",
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
            context="What is this?",
            answers="This is a test.",
            image=torch.rand(3, 224, 224),
        )

        output_sample = ImageTextSample()
        encoded_sample = self.encoder.encode(input_sample, output_sample)
        self.assertEqual(encoded_sample.__key__, input_sample.__key__)
        self.assertIsNotNone(encoded_sample.images, "The encoded sample should have images.")
        self.assertIsNotNone(encoded_sample.tokens, "The encoded sample should have tokens.")
        self.assertIsNotNone(encoded_sample.labels, "The encoded sample should have labels.")
        self.assertIsNotNone(encoded_sample.loss_mask, "The encoded sample should have a loss mask.")
        self.assertEqual(encoded_sample.images.shape, (3, 336, 336))
        self.assertEqual(
            len(encoded_sample.tokens), len(encoded_sample.labels), "Tokens and labels should have the same length."
        )
        self.assertEqual(
            len(encoded_sample.tokens),
            len(encoded_sample.loss_mask),
            "Tokens and loss mask should have the same length.",
        )

    class TestInterleavedSampleEncoder(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            cls.tokenizer = cls.processor.tokenizer
            cls.image_processor = cls.processor.image_processor

        def setUp(self):
            self.config = MultiModalSampleConfig(
                image_token=ImageToken(token_str="<image>", token_id=-200), ignore_place_holder=-100
            )
            self.encoder = InterleavedSampleEncoder(self.tokenizer, self.image_processor, self.config)

        def test_tokenize_with_text_and_images(self):
            sample_sequence = [
                "This is a test.",
                torch.rand(3, 224, 224),
                "Here is another test.",
                torch.rand(3, 224, 224),
            ]
            input_sample = InterleavedSample(__key__="sample_key", sequence=sample_sequence)
            tokens, images = self.encoder.tokenize(input_sample)
            text_tokens_part1 = self.tokenizer(sample_sequence[0], add_special_tokens=False).input_ids
            text_tokens_part2 = self.tokenizer(sample_sequence[2], add_special_tokens=False).input_ids
            expected_tokens = (
                text_tokens_part1
                + [self.config.image_token.token_id]
                + text_tokens_part2
                + [self.config.image_token.token_id]
            )
            self.assertTrue(
                torch.equal(tokens, torch.tensor(expected_tokens, dtype=torch.long)),
                msg=f"Expected tokens {expected_tokens}, but got {tokens.tolist()}",
            )
            self.assertEqual(images.shape, (1, 2, 3, 336, 336), "The concatenated image tensor shape is incorrect.")

        def test_compute_labels(self):
            tokens = torch.tensor([101, 200, 102, -200, 103, 104, -200, 105], dtype=torch.long)
            labels = self.encoder.compute_labels(tokens)
            expected_labels = torch.tensor([200, 102, -200, 103, 104, -200, 105], dtype=torch.long)
            expected_labels[expected_labels == self.config.image_token.token_id] = self.config.ignore_place_holder
            self.assertTrue(
                torch.equal(labels, expected_labels),
                msg=f"Expected labels {expected_labels.tolist()}, but got {labels.tolist()}",
            )

        def test_encode_function(self):
            sample_sequence = [
                "This is a test.",
                torch.rand(3, 224, 224),
                "Here is another test.",
                torch.rand(3, 224, 224),
            ]
            input_sample = InterleavedSample(__key__="sample_key", sequence=sample_sequence)
            output_sample = ImageTextSample()
            encoded_sample = self.encoder.encode(input_sample, output_sample)

            self.assertEqual(encoded_sample.__key__, input_sample.__key__)
            self.assertIsNotNone(encoded_sample.images, "The encoded sample should have images.")
            self.assertIsNotNone(encoded_sample.tokens, "The encoded sample should have tokens.")
            self.assertIsNotNone(encoded_sample.labels, "The encoded sample should have labels.")
            self.assertIsNotNone(encoded_sample.loss_mask, "The encoded sample should have a loss mask.")

            self.assertEqual(
                len(encoded_sample.tokens),
                len(encoded_sample.labels),
                "Tokens and labels should have the same length.",
            )
            self.assertEqual(
                len(encoded_sample.tokens),
                len(encoded_sample.loss_mask),
                "Tokens and loss mask should have the same length.",
            )

            text_tokens_part1 = self.tokenizer("This is a test.", add_special_tokens=False).input_ids
            text_tokens_part2 = self.tokenizer("Here is another test.", add_special_tokens=False).input_ids
            combined_tokens = text_tokens_part1 + [-200] + text_tokens_part2 + [-200]
            expected_labels = torch.tensor(combined_tokens[1:], dtype=torch.long)
            expected_labels[expected_labels == self.config.image_token.token_id] = self.config.ignore_place_holder
            self.assertTrue(
                torch.equal(encoded_sample.labels, expected_labels),
                msg=f"Expected labels {expected_labels.tolist()}, but got {encoded_sample.labels.tolist()}",
            )


class TestSimilarityInterleavedEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        cls.tokenizer = cls.processor.tokenizer
        cls.image_processor = cls.processor.image_processor

    def setUp(self):
        self.config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200),
            ignore_place_holder=-100,
            image_following_text=True,
        )
        self.encoder = SimilarityInterleavedEncoder(self.tokenizer, self.image_processor, self.config)

    def test_tokenize_with_interleaving_images_after_text(self):
        sample = SimilarityInterleavedSample(
            __key__="sample_key",
            images=[torch.rand(3, 224, 224), torch.rand(3, 224, 224)],
            texts=["This is the first sentence.", "This is the second sentence.", "This is the third sentence."],
            similarity_matrix=None,  # Not used in this method directly
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
        )
        sample.matched_text_indices = [0, 2]  # Images should be interleaved after texts at indices 0 and 2
        tokens, images = self.encoder.tokenize(sample)

        text_tokens_1 = self.tokenizer("This is the first sentence.", add_special_tokens=False).input_ids
        text_tokens_2 = self.tokenizer("This is the second sentence.", add_special_tokens=False).input_ids
        text_tokens_3 = self.tokenizer("This is the third sentence.", add_special_tokens=False).input_ids

        expected_tokens = text_tokens_1 + [-200] + text_tokens_2 + text_tokens_3 + [-200]
        self.assertTrue(
            torch.equal(tokens, torch.tensor(expected_tokens, dtype=torch.long)),
            msg=f"Expected tokens {expected_tokens}, but got {tokens.tolist()}",
        )
        self.assertEqual(images.shape, (1, 2, 3, 336, 336), "The concatenated image tensor shape is incorrect.")

    def test_tokenize_with_interleaving_images_before_text(self):
        self.config.image_following_text = False
        self.encoder = SimilarityInterleavedEncoder(self.tokenizer, self.image_processor, self.config)
        sample = SimilarityInterleavedSample(
            __key__="sample_key",
            images=[torch.rand(3, 224, 224), torch.rand(3, 224, 224)],  # Mock image tensors
            texts=["This is the first sentence.", "This is the second sentence.", "This is the third sentence."],
            similarity_matrix=None,  # Not used in this method directly
            __restore_key__=None,
            __subflavor__=None,
            __subflavors__=[],
        )
        sample.matched_text_indices = [0, 2]  # Images should be interleaved before texts at indices 0 and 2
        tokens, images = self.encoder.tokenize(sample)
        text_tokens_1 = self.tokenizer("This is the first sentence.", add_special_tokens=False).input_ids
        text_tokens_2 = self.tokenizer("This is the second sentence.", add_special_tokens=False).input_ids
        text_tokens_3 = self.tokenizer("This is the third sentence.", add_special_tokens=False).input_ids
        expected_tokens = [-200] + text_tokens_1 + text_tokens_2 + [-200] + text_tokens_3

        self.assertTrue(
            torch.equal(tokens, torch.tensor(expected_tokens, dtype=torch.long)),
            msg=f"Expected tokens {expected_tokens}, but got {tokens.tolist()}",
        )

        self.assertEqual(images.shape, (1, 2, 3, 336, 336), "The concatenated image tensor shape is incorrect.")


if __name__ == '__main__':
    unittest.main()
