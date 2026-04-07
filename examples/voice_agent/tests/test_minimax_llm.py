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

"""Unit tests for MiniMaxService LLM provider."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the local NeMo directory to Python path to use development version
nemo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(nemo_root))

# Mock heavy dependencies before importing llm module
sys.modules["vllm"] = MagicMock()
sys.modules["vllm.config"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["psutil"] = MagicMock()

from omegaconf import OmegaConf

from nemo.agents.voice_agent.pipecat.services.nemo.llm import MiniMaxService, get_llm_service_from_config


class TestMiniMaxService:
    """Tests for MiniMaxService."""

    def test_instantiation_with_api_key(self):
        """MiniMaxService can be created with an explicit api_key."""
        with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None):
            svc = MiniMaxService(model="MiniMax-M2.7", api_key="test-key-123")
        assert svc is not None

    def test_instantiation_uses_env_var(self):
        """MiniMaxService reads MINIMAX_API_KEY from environment when api_key is not provided."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key-456"}):
            with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None):
                svc = MiniMaxService(model="MiniMax-M2.7")
        assert svc is not None

    def test_missing_api_key_raises(self):
        """MiniMaxService raises ValueError when no API key is available."""
        env_without_key = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                MiniMaxService(model="MiniMax-M2.7")

    def test_default_base_url(self):
        """MiniMaxService uses the MiniMax overseas API base URL by default."""
        assert MiniMaxService.DEFAULT_BASE_URL == "https://api.minimax.io/v1"

    def test_custom_base_url(self):
        """MiniMaxService accepts a custom base_url."""
        custom_url = "https://custom.minimax.io/v1"
        with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None) as mock_init:
            MiniMaxService(model="MiniMax-M2.7", api_key="test-key", base_url=custom_url)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get("base_url") == custom_url

    def test_supported_models_list(self):
        """MiniMaxService exposes the two supported models."""
        assert "MiniMax-M2.7" in MiniMaxService.SUPPORTED_MODELS
        assert "MiniMax-M2.7-highspeed" in MiniMaxService.SUPPORTED_MODELS
        assert len(MiniMaxService.SUPPORTED_MODELS) == 2

    def test_api_key_passed_to_super(self):
        """The resolved API key is forwarded to OpenAILLMService."""
        with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None) as mock_init:
            MiniMaxService(model="MiniMax-M2.7", api_key="my-secret-key")
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get("api_key") == "my-secret-key"

    def test_default_model_is_m27(self):
        """MiniMaxService defaults to MiniMax-M2.7 model."""
        with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None) as mock_init:
            with patch.dict(os.environ, {"MINIMAX_API_KEY": "key"}):
                MiniMaxService()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs.get("model") == "MiniMax-M2.7"

    def test_highspeed_model(self):
        """MiniMaxService accepts MiniMax-M2.7-highspeed model."""
        with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None) as mock_init:
            MiniMaxService(model="MiniMax-M2.7-highspeed", api_key="key")
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get("model") == "MiniMax-M2.7-highspeed"


class TestGetLLMServiceFromConfigMiniMax:
    """Tests for get_llm_service_from_config with minimax backend."""

    def test_factory_creates_minimax_service(self):
        """get_llm_service_from_config returns a MiniMaxService for type=minimax."""
        cfg = OmegaConf.create(
            {
                "type": "minimax",
                "model": "MiniMax-M2.7",
                "api_key": "test-factory-key",
            }
        )
        with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None):
            svc = get_llm_service_from_config(cfg)
        assert isinstance(svc, MiniMaxService)

    def test_factory_minimax_uses_env_key(self):
        """Factory resolves MINIMAX_API_KEY when api_key is not in config."""
        cfg = OmegaConf.create({"type": "minimax", "model": "MiniMax-M2.7"})
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None):
                svc = get_llm_service_from_config(cfg)
        assert isinstance(svc, MiniMaxService)

    def test_factory_invalid_backend_raises(self):
        """get_llm_service_from_config raises AssertionError for unknown backend."""
        cfg = OmegaConf.create({"type": "unknown_backend", "model": "some-model"})
        with pytest.raises(AssertionError):
            get_llm_service_from_config(cfg)

    def test_factory_minimax_highspeed_model(self):
        """Factory correctly passes MiniMax-M2.7-highspeed model to MiniMaxService."""
        cfg = OmegaConf.create(
            {
                "type": "minimax",
                "model": "MiniMax-M2.7-highspeed",
                "api_key": "key",
            }
        )
        with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None) as mock_init:
            svc = get_llm_service_from_config(cfg)
        assert isinstance(svc, MiniMaxService)
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs.get("model") == "MiniMax-M2.7-highspeed"

    def test_factory_minimax_custom_base_url(self):
        """Factory forwards a custom base_url to MiniMaxService."""
        cfg = OmegaConf.create(
            {
                "type": "minimax",
                "model": "MiniMax-M2.7",
                "api_key": "key",
                "base_url": "https://custom.minimax.io/v1",
            }
        )
        with patch("pipecat.services.openai.llm.OpenAILLMService.__init__", return_value=None) as mock_init:
            svc = get_llm_service_from_config(cfg)
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs.get("base_url") == "https://custom.minimax.io/v1"
