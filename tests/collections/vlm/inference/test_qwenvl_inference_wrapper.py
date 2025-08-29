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

from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference_params import InferenceParams

from nemo.collections.vlm.inference.qwenvl_inference_wrapper import QwenVLInferenceWrapper


class TestQwenVLInferenceWrapper:
    """Test cases for QwenVLInferenceWrapper class"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_model = Mock()
        self.mock_config = Mock(spec=InferenceWrapperConfig)
        self.mock_config.hidden_size = 4096
        self.mock_config.params_dtype = torch.float16
        self.mock_config.inference_batch_times_seqlen_threshold = 1000
        self.mock_config.padded_vocab_size = 32000

        # Mock the parent class __init__ method
        with patch.object(QwenVLInferenceWrapper.__bases__[0], '__init__'):
            self.wrapper = QwenVLInferenceWrapper(self.mock_model, self.mock_config)
            # Manually set attributes that would normally be set by parent class
            self.wrapper.model = self.mock_model
            self.wrapper.inference_wrapper_config = self.mock_config

    def test_init(self):
        """Test QwenVLInferenceWrapper initialization"""
        assert self.wrapper.model == self.mock_model
        assert self.wrapper.inference_wrapper_config == self.mock_config

    def test_prep_inference_input_with_image(self):
        """Test prep_inference_input method with image data"""
        # Create mock input tensors
        batch_size, seq_length = 2, 10
        prompts_tokens = torch.randint(0, 1000, (batch_size, seq_length))

        # Create mock image data
        image_dict = [
            {'pixel_values': torch.randn(2, 3, 224, 224), 'image_grid_thw': torch.tensor([[1, 1, 1], [1, 1, 1]])}
        ]

        # Mock cuda method
        with patch.object(torch.Tensor, 'cuda') as mock_cuda:
            mock_cuda.return_value = torch.randn(2, 3, 224, 224)

            result = self.wrapper.prep_inference_input(prompts_tokens, image_dict)

            # Verify inference_params was set correctly
            assert hasattr(self.wrapper, 'inference_params')
            assert isinstance(self.wrapper.inference_params, InferenceParams)

            # Verify return dictionary structure
            assert 'input_ids' in result
            assert 'pixel_values' in result
            assert 'image_grid_thw' in result

            # Verify input_ids
            assert torch.equal(result['input_ids'], prompts_tokens)

            # Verify cuda was called on image tensors
            assert mock_cuda.call_count == 2

    def test_prep_inference_input_empty_image_dict(self):
        """Test prep_inference_input method with empty image_dict"""
        batch_size, seq_length = 1, 5
        prompts_tokens = torch.randint(0, 1000, (batch_size, seq_length))

        with pytest.raises(IndexError):
            self.wrapper.prep_inference_input(prompts_tokens, [])

    def test_prep_inference_input_missing_image_keys(self):
        """Test prep_inference_input method with missing image keys"""
        batch_size, seq_length = 1, 5
        prompts_tokens = torch.randint(0, 1000, (batch_size, seq_length))

        # Missing 'pixel_values' key
        image_dict = [{'image_grid_thw': torch.tensor([[1, 1, 1]])}]

        with pytest.raises(KeyError):
            self.wrapper.prep_inference_input(prompts_tokens, image_dict)

    def test_get_batch_for_context_window(self):
        """Test get_batch_for_context_window method"""
        # Create mock inference input
        inference_input = {
            'input_ids': torch.randint(0, 1000, (2, 20)),
            'pixel_values': torch.randn(2, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        context_start_position = 5
        context_end_position = 15

        result = self.wrapper.get_batch_for_context_window(
            inference_input, context_start_position, context_end_position
        )

        # Verify return dictionary structure
        assert 'input_ids' in result
        assert 'pixel_values' in result
        assert 'image_grid_thw' in result

        # Verify input_ids are truncated correctly
        expected_tokens = inference_input['input_ids'][:, :context_end_position]
        assert torch.equal(result['input_ids'], expected_tokens)

        # Verify other fields remain unchanged
        assert torch.equal(result['pixel_values'], inference_input['pixel_values'])
        assert torch.equal(result['image_grid_thw'], inference_input['image_grid_thw'])

    def test_get_batch_for_context_window_edge_cases(self):
        """Test get_batch_for_context_window method with edge cases"""
        inference_input = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 1, 1]]),
        }

        # Test with context_end_position at sequence boundary
        result = self.wrapper.get_batch_for_context_window(inference_input, 0, 10)
        assert result['input_ids'].size(1) == 10

        # Test with context_end_position beyond sequence length
        result = self.wrapper.get_batch_for_context_window(inference_input, 0, 15)
        assert result['input_ids'].size(1) == 10

        # Test with context_start_position equal to context_end_position
        result = self.wrapper.get_batch_for_context_window(inference_input, 5, 5)
        assert result['input_ids'].size(1) == 5  # Returns tokens from 0 to 5

        # Test with context_end_position = 0 (should return 0 tokens)
        result = self.wrapper.get_batch_for_context_window(inference_input, 0, 0)
        assert result['input_ids'].size(1) == 0  # Returns no tokens

    def test_forward_pass_without_pipeline_parallel(self):
        """Test forward_pass_without_pipeline_parallel method"""
        # Create mock inference input
        inference_input = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'pixel_values': torch.randn(2, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        # Mock model forward pass
        expected_logits = torch.randn(2, 10, 32000)
        self.mock_model.return_value = expected_logits

        result = self.wrapper.forward_pass_without_pipeline_parallel(inference_input)

        # Verify model was called with correct arguments
        self.mock_model.assert_called_once_with(
            attention_mask=None,
            input_ids=inference_input['input_ids'],
            pixel_values=inference_input['pixel_values'],
            image_grid_thw=inference_input['image_grid_thw'],
        )

        # Verify return value
        assert torch.equal(result, expected_logits)

    def test_forward_pass_without_pipeline_parallel_empty_input(self):
        """Test forward_pass_without_pipeline_parallel method with empty input"""
        inference_input = {}

        # Mock model forward pass
        expected_logits = torch.randn(1, 1, 32000)
        self.mock_model.return_value = expected_logits

        result = self.wrapper.forward_pass_without_pipeline_parallel(inference_input)

        # Verify model was called with only attention_mask=None
        self.mock_model.assert_called_once_with(attention_mask=None)

        # Verify return value
        assert torch.equal(result, expected_logits)

    def test_forward_pass_without_pipeline_parallel_model_error(self):
        """Test forward_pass_without_pipeline_parallel method when model raises error"""
        inference_input = {
            'input_ids': torch.randint(0, 1000, (1, 5)),
            'pixel_values': torch.randn(1, 3, 224, 224),
            'image_grid_thw': torch.tensor([[1, 1, 1]]),
        }

        # Mock model to raise an error
        self.mock_model.side_effect = RuntimeError("Model error")

        with pytest.raises(RuntimeError, match="Model error"):
            self.wrapper.forward_pass_without_pipeline_parallel(inference_input)

    def test_integration_prep_and_forward(self):
        """Test integration between prep_inference_input and forward_pass_without_pipeline_parallel"""
        # Create mock input
        batch_size, seq_length = 1, 8
        prompts_tokens = torch.randint(0, 1000, (batch_size, seq_length))
        image_dict = [{'pixel_values': torch.randn(1, 3, 224, 224), 'image_grid_thw': torch.tensor([[1, 1, 1]])}]

        # Mock cuda method
        with patch.object(torch.Tensor, 'cuda') as mock_cuda:
            mock_cuda.return_value = torch.randn(1, 3, 224, 224)

            # Prepare inference input
            inference_input = self.wrapper.prep_inference_input(prompts_tokens, image_dict)

            # Mock model forward pass
            expected_logits = torch.randn(batch_size, seq_length, 32000)
            self.mock_model.return_value = expected_logits

            # Run forward pass
            result = self.wrapper.forward_pass_without_pipeline_parallel(inference_input)

            # Verify the complete flow works
            assert torch.equal(result, expected_logits)
            assert hasattr(self.wrapper, 'inference_params')

    def test_tensor_dimensions_validation(self):
        """Test that tensor dimensions are handled correctly"""
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            seq_length = 10
            prompts_tokens = torch.randint(0, 1000, (batch_size, seq_length))
            image_dict = [
                {
                    'pixel_values': torch.randn(batch_size, 3, 224, 224),
                    'image_grid_thw': torch.tensor([[1, 1, 1]] * batch_size),
                }
            ]

            with patch.object(torch.Tensor, 'cuda') as mock_cuda:
                mock_cuda.return_value = torch.randn(batch_size, 3, 224, 224)

                result = self.wrapper.prep_inference_input(prompts_tokens, image_dict)

                # Verify inference_params was set
                assert hasattr(self.wrapper, 'inference_params')

                # Verify tensor dimensions in result
                assert result['input_ids'].size(0) == batch_size
                assert result['input_ids'].size(1) == seq_length
                assert result['pixel_values'].size(0) == batch_size

    def test_cuda_non_blocking_flag(self):
        """Test that cuda is called with non_blocking=True flag"""
        batch_size, seq_length = 1, 5
        prompts_tokens = torch.randint(0, 1000, (batch_size, seq_length))
        image_dict = [{'pixel_values': torch.randn(1, 3, 224, 224), 'image_grid_thw': torch.tensor([[1, 1, 1]])}]

        # Mock cuda method to capture the call arguments
        mock_pixel_values = Mock()
        mock_image_grid_thw = Mock()

        with patch.object(torch.Tensor, 'cuda') as mock_cuda:
            mock_cuda.return_value = torch.randn(1, 3, 224, 224)

            # Replace the tensors with our mock objects
            image_dict[0]['pixel_values'] = mock_pixel_values
            image_dict[0]['image_grid_thw'] = mock_image_grid_thw

            self.wrapper.prep_inference_input(prompts_tokens, image_dict)

            # Verify cuda was called with non_blocking=True
            mock_pixel_values.cuda.assert_called_once_with(non_blocking=True)
            mock_image_grid_thw.cuda.assert_called_once_with(non_blocking=True)


if __name__ == "__main__":
    pytest.main([__file__])
