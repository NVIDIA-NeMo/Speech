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
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from PIL import Image


class TestSetupTrainerAndRestoreModel:
    """Test cases for _setup_trainer_and_restore_model function"""

    @patch('nemo.collections.vlm.modelopt.set_modelopt_spec_if_exists_in_ckpt')
    def test_setup_trainer_and_restore_model(self, mock_set_modelopt):
        """Test _setup_trainer_and_restore_model function"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import _setup_trainer_and_restore_model

        # Mock inputs
        mock_path = "/path/to/checkpoint"
        mock_trainer = Mock()
        mock_model = Mock()
        mock_fabric = Mock()
        mock_trainer.to_fabric.return_value = mock_fabric
        mock_fabric.load_model.return_value = mock_model

        # Call function
        result = _setup_trainer_and_restore_model(mock_path, mock_trainer, mock_model)

        # Verify set_modelopt_spec_if_exists_in_ckpt was called
        mock_set_modelopt.assert_called_once_with(mock_model, mock_path)

        # Verify trainer.to_fabric was called
        mock_trainer.to_fabric.assert_called_once()

        # Verify fabric.load_model was called
        mock_fabric.load_model.assert_called_once_with(mock_path, mock_model)

        # Verify return value
        assert result == mock_model


class TestSetupInferenceWrapper:
    """Test cases for setup_inference_wrapper function"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.vocab_size = 32000

        # Mock model.module.module
        self.mock_mcore_model = Mock()
        self.mock_model.module.module = self.mock_mcore_model

        # Mock cuda and to methods
        self.mock_mcore_model.cuda.return_value = self.mock_mcore_model
        self.mock_mcore_model.to.return_value = self.mock_mcore_model

    @patch('nemo.collections.vlm.inference.base.MllamaInferenceWrapper')
    @patch('nemo.collections.vlm.inference.base.InferenceWrapperConfig')
    def test_setup_inference_wrapper_mllama(self, mock_config_cls, mock_wrapper_cls):
        """Test setup_inference_wrapper with MLlamaModelConfig"""
        # Import here to avoid import issues during test discovery
        # Mock config - create a real instance of MLlamaModelConfig
        from nemo.collections.vlm import MLlamaModelConfig
        from nemo.collections.vlm.inference.base import setup_inference_wrapper

        mock_config = Mock(spec=MLlamaModelConfig)
        mock_config.language_model_config.hidden_size = 4096
        self.mock_model.config = mock_config

        # Mock wrapper instance
        mock_wrapper_instance = Mock()
        mock_wrapper_cls.return_value = mock_wrapper_instance

        # Mock config instance
        mock_config_instance = Mock()
        mock_config_cls.return_value = mock_config_instance

        # Call function
        result = setup_inference_wrapper(self.mock_model, self.mock_tokenizer, torch.float16, 2000)

        # Verify mcore model setup
        self.mock_mcore_model.cuda.assert_called_once()
        self.mock_mcore_model.to.assert_called_once_with(torch.float16)
        self.mock_mcore_model.eval.assert_called_once()

        # Verify wrapper creation
        mock_wrapper_cls.assert_called_once_with(self.mock_mcore_model, mock_config_instance)

        # Verify config creation
        mock_config_cls.assert_called_once_with(
            hidden_size=4096,
            params_dtype=torch.float16,
            inference_batch_times_seqlen_threshold=2000,
            padded_vocab_size=32000,
        )

        # Verify return value
        assert result == mock_wrapper_instance

    @patch('nemo.collections.vlm.inference.base.LlavaInferenceWrapper')
    @patch('nemo.collections.vlm.inference.base.InferenceWrapperConfig')
    def test_setup_inference_wrapper_llava(self, mock_config_cls, mock_wrapper_cls):
        """Test setup_inference_wrapper with LlavaConfig"""
        # Import here to avoid import issues during test discovery
        # Mock config - create a real instance of LlavaConfig
        from nemo.collections.vlm import LlavaConfig
        from nemo.collections.vlm.inference.base import setup_inference_wrapper

        mock_config = Mock(spec=LlavaConfig)
        mock_config.language_transformer_config.hidden_size = 4096
        self.mock_model.config = mock_config

        # Mock wrapper instance
        mock_wrapper_instance = Mock()
        mock_wrapper_cls.return_value = mock_wrapper_instance

        # Mock config instance
        mock_config_instance = Mock()
        mock_config_cls.return_value = mock_config_instance

        # Call function
        result = setup_inference_wrapper(self.mock_model, self.mock_tokenizer, torch.bfloat16, 1000)

        # Verify wrapper creation
        mock_wrapper_cls.assert_called_once_with(self.mock_mcore_model, mock_config_instance)

        # Verify config creation
        mock_config_cls.assert_called_once_with(
            hidden_size=4096,
            params_dtype=torch.bfloat16,
            inference_batch_times_seqlen_threshold=1000,
            padded_vocab_size=32000,
        )

        # Verify return value
        assert result == mock_wrapper_instance

    @patch('nemo.collections.vlm.inference.base.QwenVLInferenceWrapper')
    @patch('nemo.collections.vlm.inference.base.InferenceWrapperConfig')
    def test_setup_inference_wrapper_qwenvl(self, mock_config_cls, mock_wrapper_cls):
        """Test setup_inference_wrapper with Qwen2VLConfig"""
        # Import here to avoid import issues during test discovery
        # Mock config - create a real instance of Qwen2VLConfig
        from nemo.collections.vlm import Qwen2VLConfig
        from nemo.collections.vlm.inference.base import setup_inference_wrapper

        mock_config = Mock(spec=Qwen2VLConfig)
        mock_config.language_transformer_config.hidden_size = 4096
        self.mock_model.config = mock_config

        # Mock wrapper instance
        mock_wrapper_instance = Mock()
        mock_wrapper_cls.return_value = mock_wrapper_instance

        # Mock config instance
        mock_config_instance = Mock()
        mock_config_cls.return_value = mock_config_instance

        # Call function
        result = setup_inference_wrapper(self.mock_model, self.mock_tokenizer, torch.bfloat16, 1500)

        # Verify wrapper creation
        mock_wrapper_cls.assert_called_once_with(self.mock_mcore_model, mock_config_instance)

        # Verify config creation
        mock_config_cls.assert_called_once_with(
            hidden_size=4096,
            params_dtype=torch.bfloat16,
            inference_batch_times_seqlen_threshold=1500,
            padded_vocab_size=32000,
        )

        # Verify return value
        assert result == mock_wrapper_instance

    def test_setup_inference_wrapper_unknown_config(self):
        """Test setup_inference_wrapper with unknown config type"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import setup_inference_wrapper

        # Mock unknown config
        mock_config = Mock()
        self.mock_model.config = mock_config

        # Call function and expect ValueError
        with pytest.raises(ValueError, match="Unknown model config"):
            setup_inference_wrapper(self.mock_model, self.mock_tokenizer)


class TestSetupModelAndTokenizer:
    """Test cases for setup_model_and_tokenizer function"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_path = "/path/to/checkpoint"
        self.mock_trainer = Mock()

    @patch('nemo.collections.vlm.inference.base.nl.io.load_context')
    @patch('nemo.collections.vlm.inference.base.AutoProcessor.from_pretrained')
    @patch('nemo.collections.vlm.inference.base.vlm.MLlamaModel')
    @patch('nemo.collections.vlm.inference.base._setup_trainer_and_restore_model')
    @patch('nemo.collections.vlm.inference.base.setup_inference_wrapper')
    def test_setup_model_and_tokenizer_mllama(
        self, mock_setup_wrapper, mock_setup_trainer, mock_model_cls, mock_processor_cls, mock_load_context
    ):
        """Test setup_model_and_tokenizer with MLlamaModelConfig"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import setup_model_and_tokenizer

        # Mock context and config
        mock_context = Mock()
        from nemo.collections.vlm import MLlamaModelConfig

        mock_config = Mock(spec=MLlamaModelConfig)
        mock_load_context.return_value = mock_context
        mock_context.config = mock_config

        # Mock processor and model
        mock_processor = Mock()
        mock_processor.tokenizer = self.mock_tokenizer = Mock()
        mock_processor_cls.return_value = mock_processor

        mock_model = Mock()
        mock_model_cls.return_value = mock_model

        # Mock setup functions
        mock_setup_trainer.return_value = mock_model
        mock_wrapper = Mock()
        mock_setup_wrapper.return_value = mock_wrapper

        # Call function
        result_wrapper, result_processor = setup_model_and_tokenizer(
            self.mock_path,
            self.mock_trainer,
            tp_size=2,
            pp_size=1,
            params_dtype=torch.float16,
            inference_batch_times_seqlen_threshold=2000,
        )

        # Verify context loading
        mock_load_context.assert_called_once()

        # Verify processor creation
        mock_processor_cls.assert_called_once_with("meta-llama/Llama-3.2-11B-Vision-Instruct")

        # Verify model creation
        mock_model_cls.assert_called_once_with(mock_config, tokenizer=mock_processor.tokenizer)

        # Verify setup calls
        mock_setup_trainer.assert_called_once_with(path=self.mock_path, trainer=self.mock_trainer, model=mock_model)

        mock_setup_wrapper.assert_called_once_with(mock_model, mock_processor.tokenizer, torch.float16, 2000)

        # Verify return values
        assert result_wrapper == mock_wrapper
        assert result_processor == mock_processor

    @patch('nemo.collections.vlm.inference.base.nl.io.load_context')
    @patch('nemo.collections.vlm.inference.base.AutoProcessor.from_pretrained')
    @patch('nemo.collections.vlm.inference.base.vlm.LlavaModel')
    @patch('nemo.collections.vlm.inference.base._setup_trainer_and_restore_model')
    @patch('nemo.collections.vlm.inference.base.setup_inference_wrapper')
    def test_setup_model_and_tokenizer_llava(
        self, mock_setup_wrapper, mock_setup_trainer, mock_model_cls, mock_processor_cls, mock_load_context
    ):
        """Test setup_model_and_tokenizer with LlavaConfig"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import setup_model_and_tokenizer

        # Mock context and config
        mock_context = Mock()
        from nemo.collections.vlm import LlavaConfig

        mock_config = Mock(spec=LlavaConfig)
        mock_load_context.return_value = mock_context
        mock_context.config = mock_config

        # Mock processor and model
        mock_processor = Mock()
        mock_processor.tokenizer = self.mock_tokenizer = Mock()
        mock_processor_cls.return_value = mock_processor

        mock_model = Mock()
        mock_model_cls.return_value = mock_model

        # Mock setup functions
        mock_setup_trainer.return_value = mock_model
        mock_wrapper = Mock()
        mock_setup_wrapper.return_value = mock_wrapper

        # Call function
        result_wrapper, result_processor = setup_model_and_tokenizer(self.mock_path, self.mock_trainer)

        # Verify processor creation
        mock_processor_cls.assert_called_once_with("llava-hf/llava-1.5-7b-hf")

        # Verify model creation
        mock_model_cls.assert_called_once_with(mock_config, tokenizer=mock_processor.tokenizer)

        # Verify return values
        assert result_wrapper == mock_wrapper
        assert result_processor == mock_processor

    @patch('nemo.collections.vlm.inference.base.nl.io.load_context')
    @patch('nemo.collections.vlm.inference.base.AutoProcessor.from_pretrained')
    @patch('nemo.collections.vlm.inference.base.vlm.Qwen2VLModel')
    @patch('nemo.collections.vlm.inference.base._setup_trainer_and_restore_model')
    @patch('nemo.collections.vlm.inference.base.setup_inference_wrapper')
    def test_setup_model_and_tokenizer_qwenvl_3b(
        self, mock_setup_wrapper, mock_setup_trainer, mock_model_cls, mock_processor_cls, mock_load_context
    ):
        """Test setup_model_and_tokenizer with Qwen2VLConfig (3B model)"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import setup_model_and_tokenizer

        # Mock context and config
        mock_context = Mock()
        from nemo.collections.vlm import Qwen2VLConfig

        mock_config = Mock(spec=Qwen2VLConfig)
        mock_config.vision_projection_config.projector_type = "mcore_mlp"
        mock_config.vision_projection_config.hidden_size = 2048
        mock_load_context.return_value = mock_context
        mock_context.config = mock_config

        # Mock processor and model
        mock_processor = Mock()
        mock_processor.tokenizer = self.mock_tokenizer = Mock()
        mock_processor_cls.return_value = mock_processor

        mock_model = Mock()
        mock_model_cls.return_value = mock_model

        # Mock setup functions
        mock_setup_trainer.return_value = mock_model
        mock_wrapper = Mock()
        mock_setup_wrapper.return_value = mock_wrapper

        # Call function
        result_wrapper, result_processor = setup_model_and_tokenizer(self.mock_path, self.mock_trainer)

        # Verify processor creation with correct parameters
        mock_processor_cls.assert_called_once_with("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=12544, max_pixels=50176)

        # Verify model creation
        mock_model_cls.assert_called_once_with(
            mock_config, tokenizer=mock_processor.tokenizer, model_version="qwen25-vl"
        )

        # Verify return values
        assert result_wrapper == mock_wrapper
        assert result_processor == mock_processor

    @patch('nemo.collections.vlm.inference.base.nl.io.load_context')
    def test_setup_model_and_tokenizer_qwenvl_invalid_projector(self, mock_load_context):
        """Test setup_model_and_tokenizer with Qwen2VLConfig invalid projector type"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import setup_model_and_tokenizer

        # Mock context and config
        mock_context = Mock()
        from nemo.collections.vlm import Qwen2VLConfig

        mock_config = Mock(spec=Qwen2VLConfig)
        mock_config.vision_projection_config.projector_type = "invalid"
        mock_load_context.return_value = mock_context
        mock_context.config = mock_config

        # Call function and expect ValueError
        with pytest.raises(ValueError, match="Only support Qwen2.5-VL with mcore_mlp projector type"):
            setup_model_and_tokenizer(self.mock_path, self.mock_trainer)

    @patch('nemo.collections.vlm.inference.base.nl.io.load_context')
    def test_setup_model_and_tokenizer_qwenvl_unknown_size(self, mock_load_context):
        """Test setup_model_and_tokenizer with Qwen2VLConfig unknown hidden size"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import setup_model_and_tokenizer

        # Mock context and config
        mock_context = Mock()
        from nemo.collections.vlm import Qwen2VLConfig

        mock_config = Mock(spec=Qwen2VLConfig)
        mock_config.vision_projection_config.projector_type = "mcore_mlp"
        mock_config.vision_projection_config.hidden_size = 9999  # Unknown size
        mock_load_context.return_value = mock_context
        mock_context.config = mock_config

        # Call function and expect ValueError
        with pytest.raises(ValueError, match="Unknown model size"):
            setup_model_and_tokenizer(self.mock_path, self.mock_trainer)

    @patch('nemo.collections.vlm.inference.base.nl.io.load_context')
    def test_setup_model_and_tokenizer_unknown_config(self, mock_load_context):
        """Test setup_model_and_tokenizer with unknown config type"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import setup_model_and_tokenizer

        # Mock context and config
        mock_context = Mock()

        # Use a different class that won't match any of the expected config types
        class UnknownConfig:
            pass

        mock_config = Mock(spec=UnknownConfig)
        mock_load_context.return_value = mock_context
        mock_context.config = mock_config

        # Call function and expect ValueError
        with pytest.raises(ValueError, match="Unknown model config"):
            setup_model_and_tokenizer(self.mock_path, self.mock_trainer)

    @patch('nemo.collections.vlm.inference.base.nl.io.load_context')
    @patch('nemo.collections.vlm.inference.base.AutoProcessor.from_pretrained')
    @patch('nemo.collections.vlm.inference.base.vlm.MLlamaModel')
    @patch('nemo.collections.vlm.inference.base.nl.MegatronStrategy')
    @patch('nemo.collections.vlm.inference.base.nl.Trainer')
    @patch('nemo.collections.vlm.inference.base._setup_trainer_and_restore_model')
    @patch('nemo.collections.vlm.inference.base.setup_inference_wrapper')
    def test_setup_model_and_tokenizer_no_trainer(
        self,
        mock_setup_wrapper,
        mock_setup_trainer,
        mock_trainer_cls,
        mock_strategy_cls,
        mock_model_cls,
        mock_processor_cls,
        mock_load_context,
    ):
        """Test setup_model_and_tokenizer without providing trainer"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import setup_model_and_tokenizer

        # Mock context and config
        mock_context = Mock()
        from nemo.collections.vlm import MLlamaModelConfig

        mock_config = Mock(spec=MLlamaModelConfig)
        mock_load_context.return_value = mock_context
        mock_context.config = mock_config

        # Mock processor and model
        mock_processor = Mock()
        mock_processor.tokenizer = self.mock_tokenizer = Mock()
        mock_processor_cls.return_value = mock_processor

        mock_model = Mock()
        mock_model_cls.return_value = mock_model

        # Mock strategy and trainer
        mock_strategy = Mock()
        mock_strategy_cls.return_value = mock_strategy

        mock_trainer = Mock()
        mock_trainer_cls.return_value = mock_trainer

        # Mock setup functions
        mock_setup_trainer.return_value = mock_model
        mock_wrapper = Mock()
        mock_setup_wrapper.return_value = mock_wrapper

        # Call function without trainer
        result_wrapper, result_processor = setup_model_and_tokenizer(
            self.mock_path, trainer=None, tp_size=2, pp_size=1
        )

        # Verify strategy creation
        mock_strategy_cls.assert_called_once_with(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1, ckpt_include_optimizer=False
        )

        # Verify trainer creation
        mock_trainer_cls.assert_called_once()

        # Verify return values
        assert result_wrapper == mock_wrapper
        assert result_processor == mock_processor


class TestGenerate:
    """Test cases for generate function"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_wrapped_model = Mock(spec=AbstractModelInferenceWrapper)
        self.mock_tokenizer = Mock()
        self.mock_image_processor = Mock()
        self.mock_prompts = ["test prompt 1", "test prompt 2"]
        self.mock_images = [Image.new('RGB', (224, 224))]
        self.mock_processor = Mock()

    @patch('nemo.collections.vlm.inference.base.QwenVLTextGenerationController')
    @patch('nemo.collections.vlm.inference.base.VLMEngine')
    @patch('nemo.collections.vlm.inference.base.CommonInferenceParams')
    def test_generate_qwenvl(self, mock_common_params_cls, mock_engine_cls, mock_controller_cls):
        """Test generate function with QwenVLInferenceWrapper"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import generate

        # Mock QwenVLInferenceWrapper - need to use real class for isinstance check
        from nemo.collections.vlm.inference.qwenvl_inference_wrapper import QwenVLInferenceWrapper

        self.mock_wrapped_model.__class__ = QwenVLInferenceWrapper

        # Mock controller - mock the class itself to skip init
        mock_controller = Mock()
        mock_controller_cls.return_value = mock_controller

        # Mock engine
        mock_engine = Mock()
        mock_engine_cls.return_value = mock_engine

        # Mock common inference params
        mock_common_params = Mock()
        mock_common_params_cls.return_value = mock_common_params

        # Mock engine results
        mock_results = ["result1", "result2"]
        mock_engine.generate.return_value = mock_results

        # Call function
        result = generate(
            self.mock_wrapped_model,
            self.mock_tokenizer,
            self.mock_image_processor,
            self.mock_prompts,
            self.mock_images,
            processor=self.mock_processor,
            max_batch_size=8,
            random_seed=42,
            inference_params=mock_common_params,
        )

        # Verify engine creation
        mock_engine_cls.assert_called_once_with(
            text_generation_controller=mock_controller, max_batch_size=8, random_seed=42
        )

        # Verify engine.generate was called
        mock_engine.generate.assert_called_once_with(
            prompts=self.mock_prompts, images=self.mock_images, common_inference_params=mock_common_params
        )

        # Verify return value
        assert result == mock_results

    @patch('nemo.collections.vlm.inference.base.VLMTextGenerationController')
    @patch('nemo.collections.vlm.inference.base.VLMEngine')
    @patch('nemo.collections.vlm.inference.base.CommonInferenceParams')
    def test_generate_other_model(self, mock_common_params_cls, mock_engine_cls, mock_controller_cls):
        """Test generate function with non-QwenVL model"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import generate

        # Mock other model type - use a different class so it goes to VLMTextGenerationController
        from nemo.collections.vlm.inference.mllama_inference_wrapper import MllamaInferenceWrapper

        self.mock_wrapped_model.__class__ = MllamaInferenceWrapper

        # Mock controller - mock the class itself to skip init
        mock_controller = Mock()
        mock_controller_cls.return_value = mock_controller

        # Mock engine
        mock_engine = Mock()
        mock_engine_cls.return_value = mock_engine

        # Mock common inference params
        mock_common_params = Mock()
        mock_common_params_cls.return_value = mock_common_params

        # Mock engine results
        mock_results = ["result1", "result2"]
        mock_engine.generate.return_value = mock_results

        # Call function
        result = generate(
            self.mock_wrapped_model,
            self.mock_tokenizer,
            self.mock_image_processor,
            self.mock_prompts,
            self.mock_images,
            max_batch_size=4,
        )

        # Verify engine creation
        mock_engine_cls.assert_called_once_with(
            text_generation_controller=mock_controller, max_batch_size=4, random_seed=None
        )

        # Verify default common inference params were created
        mock_common_params_cls.assert_called_once_with(num_tokens_to_generate=50)

        # Verify engine.generate was called
        mock_engine.generate.assert_called_once_with(
            prompts=self.mock_prompts, images=self.mock_images, common_inference_params=mock_common_params
        )

        # Verify return value
        assert result == mock_results

    @patch('nemo.collections.vlm.inference.base.QwenVLTextGenerationController')
    @patch('nemo.collections.vlm.inference.base.VLMEngine')
    @patch('nemo.collections.vlm.inference.base.CommonInferenceParams')
    def test_generate_with_defaults(self, mock_common_params_cls, mock_engine_cls, mock_controller_cls):
        """Test generate function with default parameters"""
        # Import here to avoid import issues during test discovery
        from nemo.collections.vlm.inference.base import generate

        # Mock QwenVLInferenceWrapper - need to use real class for isinstance check
        from nemo.collections.vlm.inference.qwenvl_inference_wrapper import QwenVLInferenceWrapper

        self.mock_wrapped_model.__class__ = QwenVLInferenceWrapper

        # Mock controller - mock the class itself to skip init
        mock_controller = Mock()
        mock_controller_cls.return_value = mock_controller

        # Mock engine
        mock_engine = Mock()
        mock_engine_cls.return_value = mock_engine

        # Mock common inference params
        mock_common_params = Mock()
        mock_common_params_cls.return_value = mock_common_params

        # Mock engine results
        mock_results = ["result1"]
        mock_engine.generate.return_value = mock_results

        # Call function with minimal parameters
        result = generate(
            self.mock_wrapped_model,
            self.mock_tokenizer,
            self.mock_image_processor,
            self.mock_prompts[:1],
            self.mock_images[:1],
        )

        # Verify default values were used
        mock_engine_cls.assert_called_once_with(
            text_generation_controller=mock_controller, max_batch_size=4, random_seed=None  # default  # default
        )

        # Verify default common inference params were created
        mock_common_params_cls.assert_called_once_with(num_tokens_to_generate=50)

        # Verify return value
        assert result == mock_results
