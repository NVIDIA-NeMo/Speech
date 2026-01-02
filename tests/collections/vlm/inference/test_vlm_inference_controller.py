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

from collections import OrderedDict
from unittest.mock import Mock, patch

import pytest
import torch

from nemo.collections.vlm.inference.vlm_inference_controller import (
    QwenVLTextGenerationController,
    TokenizerWrapper,
    VLMTextGenerationController,
)


class TestTokenizerWrapper:
    """Test cases for TokenizerWrapper class"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.eos_token_id = 2
        self.mock_tokenizer.vocab_size = 32000
        self.tokenizer_wrapper = TokenizerWrapper(self.mock_tokenizer)

    def test_init(self):
        """Test TokenizerWrapper initialization"""
        assert self.tokenizer_wrapper.eod == 2
        assert self.tokenizer_wrapper.vocab_size is None
        assert self.tokenizer_wrapper._tokenizer == self.mock_tokenizer

    def test_detokenize(self):
        """Test detokenize method"""
        mock_tokens = [1, 2, 3]
        expected_result = "test output"
        self.mock_tokenizer.decode.return_value = expected_result

        result = self.tokenizer_wrapper.detokenize(mock_tokens)

        self.mock_tokenizer.decode.assert_called_once_with(mock_tokens, skip_special_tokens=False)
        assert result == expected_result

    def test_tokenize(self):
        """Test tokenize method"""
        mock_prompt = "test prompt"
        expected_tokens = [1, 2, 3]
        self.mock_tokenizer.encode.return_value = expected_tokens

        result = self.tokenizer_wrapper.tokenize(mock_prompt)

        self.mock_tokenizer.encode.assert_called_once_with(mock_prompt, add_special_tokens=False)
        assert result == expected_tokens


class TestVLMTextGenerationController:
    """Test cases for VLMTextGenerationController class"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_inference_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_image_processor = Mock()
        self.mock_image_processor.size = {'height': 224, 'width': 224}

        # Mock just the parent class __init__ method to prevent real initialization
        with patch.object(VLMTextGenerationController.__bases__[0], '__init__'):
            # Create the controller
            self.controller = VLMTextGenerationController(
                self.mock_inference_model, self.mock_tokenizer, self.mock_image_processor
            )

            # Manually set attributes that would normally be set by parent class
            self.controller.tokenizer = TokenizerWrapper(self.mock_tokenizer)
            self.controller.inference_wrapped_model = self.mock_inference_model

    def test_init(self):
        """Test VLMTextGenerationController initialization"""
        assert self.controller.image_processor == self.mock_image_processor

    def test_tokenize_prompt_with_no_image(self):
        """Test tokenize_prompt method when image is None"""
        mock_prompt = "test prompt"
        mock_tokens = [1, 2, 3]
        self.mock_tokenizer.encode.return_value = mock_tokens

        tokens, image_dict = self.controller.tokenize_prompt(mock_prompt, None)

        self.mock_tokenizer.encode.assert_called_once_with(mock_prompt, add_special_tokens=False)
        assert tokens == mock_tokens
        assert 'pixel_values' in image_dict
        assert 'aspect_ratio_ids' in image_dict
        assert 'num_tiles' in image_dict
        assert image_dict['pixel_values'].shape == (1, 4, 3, 224, 224)
        assert image_dict['aspect_ratio_ids'].shape == (1,)
        assert image_dict['num_tiles'] == [0]

    def test_tokenize_prompt_with_image(self):
        """Test tokenize_prompt method when image is provided"""
        mock_prompt = "test prompt"
        mock_image = Mock()
        mock_tokens = [1, 2, 3]
        self.mock_tokenizer.encode.return_value = mock_tokens

        # Mock image processor output
        mock_processed = {
            'pixel_values': torch.randn(1, 4, 3, 224, 224),
            'aspect_ratio_ids': torch.tensor([1], dtype=torch.long),
            'num_tiles': [2],
            'other_key': 'should_be_filtered',
        }
        self.mock_image_processor.preprocess.return_value = mock_processed

        tokens, image_dict = self.controller.tokenize_prompt(mock_prompt, mock_image)

        self.mock_tokenizer.encode.assert_called_once_with(mock_prompt, add_special_tokens=False)
        self.mock_image_processor.preprocess.assert_called_once_with(mock_image, return_tensors='pt')
        assert tokens == mock_tokens
        assert 'pixel_values' in image_dict
        assert 'aspect_ratio_ids' in image_dict
        assert 'num_tiles' in image_dict
        assert 'other_key' not in image_dict
        assert image_dict['pixel_values'].shape == (4, 3, 224, 224)

    def test_prep_inference_input(self):
        """Test prep_inference_input method"""
        mock_prompts_tokens = torch.randn(2, 10)
        mock_active_requests = OrderedDict(
            [('req1', Mock(encoder_prompt='image1')), ('req2', Mock(encoder_prompt='image2'))]
        )

        expected_result = {'processed': 'data'}
        self.mock_inference_model.prep_inference_input.return_value = expected_result

        result = self.controller.prep_inference_input(
            mock_prompts_tokens, mock_active_requests, use_attention_mask=True
        )

        self.mock_inference_model.prep_inference_input.assert_called_once_with(
            prompts_tokens=mock_prompts_tokens, image_dict=['image1', 'image2']
        )
        assert result == expected_result


class TestQwenVLTextGenerationController:
    """Test cases for QwenVLTextGenerationController class"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_inference_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_image_processor = Mock()
        self.mock_image_processor.size = {'height': 224, 'width': 224}
        self.mock_processor = Mock()

        # Mock just the parent class __init__ method to prevent real initialization
        with patch.object(QwenVLTextGenerationController.__bases__[0], '__init__'):
            # Create the controller
            self.controller = QwenVLTextGenerationController(
                self.mock_inference_model, self.mock_tokenizer, self.mock_image_processor, self.mock_processor
            )

            # Manually set attributes that would normally be set by parent class
            self.controller.inference_wrapped_model = self.mock_inference_model

    def test_init(self):
        """Test QwenVLTextGenerationController initialization"""
        assert hasattr(self.controller, 'tokenizer')
        assert hasattr(self.controller, 'processor')
        assert self.controller.processor == self.mock_processor

    def test_qwenvl_tokenizer_detokenize(self):
        """Test QwenVLTokenizer detokenize method"""
        mock_tokens = [151652, 151655, -200, 100, 200]
        expected_result = "processed output"
        self.mock_tokenizer.decode.return_value = expected_result

        # Get the QwenVLTokenizer instance
        qwen_tokenizer = self.controller.tokenizer

        result = qwen_tokenizer.detokenize(mock_tokens)

        # Verify the result is correct
        assert result == expected_result
        # Verify the mock was called (the QwenVLTokenizer uses self.mock_tokenizer internally)
        self.mock_tokenizer.decode.assert_called_once()

    def test_tokenize_prompt(self):
        """Test QwenVLTextGenerationController tokenize_prompt method"""
        mock_prompt = "test prompt"
        mock_image = Mock()

        # Mock processor output
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 151655, 3, 4]]),
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([1, 1, 1]),
        }
        self.mock_processor.return_value = mock_inputs

        tokens, image_dict = self.controller.tokenize_prompt(mock_prompt, mock_image)

        self.mock_processor.assert_called_once_with(
            text=[mock_prompt], images=mock_image, padding=True, return_tensors="pt"
        )

        # Check that 151655 tokens are replaced with -200
        assert tokens == [1, 2, -200, 3, 4]
        assert 'pixel_values' in image_dict
        assert 'image_grid_thw' in image_dict
        assert image_dict['pixel_values'].shape == (1, 3, 224, 224)


class TestIntegration:
    """Integration tests for the controllers"""

    def test_vlm_controller_inheritance(self):
        """Test that VLMTextGenerationController properly inherits from parent"""
        mock_inference_model = Mock()
        mock_tokenizer = Mock()
        mock_image_processor = Mock()

        # Mock just the parent class __init__ method to prevent real initialization
        with patch.object(VLMTextGenerationController.__bases__[0], '__init__'):
            controller = VLMTextGenerationController(mock_inference_model, mock_tokenizer, mock_image_processor)

            # Manually set attributes that would normally be set by parent class
            controller.tokenizer = TokenizerWrapper(mock_tokenizer)
            controller.inference_wrapped_model = mock_inference_model

            # Verify the controller was created successfully and has expected attributes
            assert controller is not None
            assert hasattr(controller, 'image_processor')

    def test_qwenvl_controller_inheritance(self):
        """Test that QwenVLTextGenerationController properly inherits from VLMTextGenerationController"""
        mock_inference_model = Mock()
        mock_tokenizer = Mock()
        mock_image_processor = Mock()
        mock_processor = Mock()

        # Mock just the parent class __init__ method to prevent real initialization
        with patch.object(QwenVLTextGenerationController.__bases__[0], '__init__'):
            controller = QwenVLTextGenerationController(
                mock_inference_model, mock_tokenizer, mock_image_processor, mock_processor
            )

            # Manually set attributes that would normally be set by parent class
            controller.tokenizer = TokenizerWrapper(mock_tokenizer)
            controller.inference_wrapped_model = mock_inference_model

            # Verify the controller was created successfully and has expected attributes
            assert controller is not None
            assert hasattr(controller, 'processor')

    def test_tokenizer_wrapper_interface(self):
        """Test that TokenizerWrapper provides the expected interface"""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 1
        wrapper = TokenizerWrapper(mock_tokenizer)

        # Test that required attributes exist
        assert hasattr(wrapper, 'eod')
        assert hasattr(wrapper, 'vocab_size')
        assert hasattr(wrapper, '_tokenizer')
        assert hasattr(wrapper, 'detokenize')
        assert hasattr(wrapper, 'tokenize')

        # Test that methods are callable
        assert callable(wrapper.detokenize)
        assert callable(wrapper.tokenize)


if __name__ == "__main__":
    pytest.main([__file__])
