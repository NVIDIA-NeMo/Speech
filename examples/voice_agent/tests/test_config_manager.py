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

import os
import sys
from pathlib import Path

# Add the local NeMo directory to Python path to use development version
nemo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(nemo_root))

import pytest
from omegaconf import DictConfig, OmegaConf
from pipecat.audio.vad.silero import VADParams

from nemo.agents.voice_agent.pipecat.services.nemo.diar import NeMoDiarInputParams
from nemo.agents.voice_agent.pipecat.services.nemo.stt import NeMoSTTInputParams
from nemo.agents.voice_agent.utils.config_manager import ConfigManager


@pytest.fixture
def voice_agent_server_base_path():
    """Retrieve the NeMo root path from __file__ variable"""
    nemo_root_path = Path(__file__).resolve().parents[3]

    # Check if the expected directories exist in the NeMo root
    expected_dirs = ["nemo", "tests", "examples", "requirements"]
    existing_dirs = [d.name for d in nemo_root_path.iterdir() if d.is_dir()]

    if not all(sub in existing_dirs for sub in expected_dirs):
        raise ValueError(
            f"{nemo_root_path} is not a NeMo root path. Expected dirs: {expected_dirs}, Found dirs: {existing_dirs}"
        )

    voice_agent_root_path = os.path.join(nemo_root_path, "examples", "voice_agent", "server")
    return voice_agent_root_path


class TestDefaultConfigs:
    """Test suite for ConfigManager class."""

    @pytest.mark.unit
    def test_constructor_with_valid_path(self, voice_agent_server_base_path):
        """Test ConfigManager initialization with valid configuration files."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        # Verify initialization
        assert config_manager._server_base_path == voice_agent_server_base_path
        assert config_manager.SAMPLE_RATE == 16000
        assert config_manager.RAW_AUDIO_FRAME_LEN_IN_SECS == 0.016
        assert isinstance(config_manager.vad_params, VADParams)
        assert isinstance(config_manager.stt_params, NeMoSTTInputParams)
        assert isinstance(config_manager.diar_params, NeMoDiarInputParams)

    @pytest.mark.unit
    def test_constructor_with_invalid_path(self):
        """Test ConfigManager initialization with invalid path."""
        with pytest.raises(FileNotFoundError):
            ConfigManager("/nonexistent/path")

    @pytest.mark.unit
    def test_load_model_registry_success(self, voice_agent_server_base_path):
        """Test successful model registry loading."""
        config_manager = ConfigManager(voice_agent_server_base_path)

        assert config_manager.model_registry is not None
        assert "llm_models" in config_manager.model_registry
        assert "tts_models" in config_manager.model_registry
        assert "stt_models" in config_manager.model_registry

    @pytest.mark.unit
    def test_configure_stt_nemo_model(self, voice_agent_server_base_path):
        """Test STT configuration for NeMo model."""
        # Create necessary files
        config_manager = ConfigManager(voice_agent_server_base_path)

        assert "stt_en_fastconformer" in config_manager.STT_MODEL_PATH
        assert isinstance(config_manager.stt_params, NeMoSTTInputParams)

    @pytest.mark.unit
    def test_configure_stt_with_model_config(self, voice_agent_server_base_path):
        """Test STT configuration with custom model config."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        assert hasattr(config_manager, "STT_MODEL_PATH")

    @pytest.mark.unit
    def test_configure_diarization(self, voice_agent_server_base_path):
        """Test diarization configuration."""
        config_manager = ConfigManager(voice_agent_server_base_path)

        assert hasattr(config_manager, "DIAR_MODEL") and isinstance(config_manager.DIAR_MODEL, str)
        assert hasattr(config_manager, "USE_DIAR") and isinstance(config_manager.USE_DIAR, bool)
        assert isinstance(config_manager.diar_params, NeMoDiarInputParams)

    @pytest.mark.unit
    def test_configure_turn_taking(self, voice_agent_server_base_path):
        """Test turn taking configuration."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        assert hasattr(config_manager, "TURN_TAKING_BACKCHANNEL_PHRASES_PATH") and isinstance(
            config_manager.TURN_TAKING_BACKCHANNEL_PHRASES_PATH, str
        )
        assert hasattr(config_manager, "TURN_TAKING_MAX_BUFFER_SIZE") and isinstance(
            config_manager.TURN_TAKING_MAX_BUFFER_SIZE, int
        )
        assert hasattr(config_manager, "TURN_TAKING_BOT_STOP_DELAY") and isinstance(
            config_manager.TURN_TAKING_BOT_STOP_DELAY, float
        )

    @pytest.mark.unit
    def test_configure_turn_taking_backchannel_phrases(self, voice_agent_server_base_path):
        """Test turn taking configuration."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        # Load backchannel phrases yaml file
        file_path = os.path.join(
            voice_agent_server_base_path, os.path.basename(config_manager.TURN_TAKING_BACKCHANNEL_PHRASES_PATH)
        )
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            backchannel_phrases = OmegaConf.load(f)
        backchannel_phrases = list(backchannel_phrases)
        assert isinstance(backchannel_phrases, list)
        assert all(isinstance(item, str) for item in backchannel_phrases)

    @pytest.mark.unit
    def test_configure_llm_with_registry_model(self, voice_agent_server_base_path):
        """Test LLM configuration with model from registry."""
        config_manager = ConfigManager(voice_agent_server_base_path)

        assert hasattr(config_manager, "SYSTEM_ROLE") and isinstance(config_manager.SYSTEM_ROLE, str)
        assert hasattr(config_manager, "SYSTEM_PROMPT") and isinstance(config_manager.SYSTEM_PROMPT, str)

    @pytest.mark.unit
    def test_configure_llm_with_file_system_prompt(self, voice_agent_server_base_path):
        config_manager = ConfigManager(voice_agent_server_base_path)
        assert hasattr(config_manager, "SYSTEM_PROMPT") and isinstance(config_manager.SYSTEM_PROMPT, str)

    @pytest.mark.unit
    def test_configure_llm_reasoning_model(self, voice_agent_server_base_path):
        """Test LLM configuration for reasoning model."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        assert hasattr(config_manager, "SYSTEM_ROLE") and isinstance(config_manager.SYSTEM_ROLE, str)

    @pytest.mark.unit
    def test_configure_llm_fallback_to_generic(self, voice_agent_server_base_path):
        """Test LLM configuration fallback to generic HF model."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        assert hasattr(config_manager, "SYSTEM_ROLE") and isinstance(config_manager.SYSTEM_ROLE, str)

    @pytest.mark.unit
    def test_configure_tts_nemo_model(self, voice_agent_server_base_path):
        """Test TTS configuration for NeMo model."""
        config_manager = ConfigManager(voice_agent_server_base_path)

        assert hasattr(config_manager, "TTS_MAIN_MODEL_ID")
        assert hasattr(config_manager, "TTS_SUB_MODEL_ID")
        assert hasattr(config_manager, "TTS_DEVICE")

    @pytest.mark.unit
    def test_configure_tts_with_optional_params(self, voice_agent_server_base_path):
        """Test TTS configuration with optional parameters."""
        config_manager = ConfigManager(voice_agent_server_base_path)

        assert hasattr(config_manager, "TTS_THINK_TOKENS") and isinstance(config_manager.TTS_THINK_TOKENS, list)
        assert all(isinstance(item, str) for item in config_manager.TTS_THINK_TOKENS)
        assert hasattr(config_manager, "TTS_EXTRA_SEPARATOR") and isinstance(config_manager.TTS_EXTRA_SEPARATOR, list)
        assert all(isinstance(item, str) for item in config_manager.TTS_EXTRA_SEPARATOR)

    @pytest.mark.unit
    def test_get_server_config(self, voice_agent_server_base_path):
        """Test get_server_config method."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        server_config = config_manager.get_server_config()

        assert isinstance(server_config, DictConfig)
        assert hasattr(server_config.transport, "audio_out_10ms_chunks")
        assert isinstance(server_config.transport.audio_out_10ms_chunks, int)

    @pytest.mark.unit
    def test_get_model_registry(self, voice_agent_server_base_path):
        """Test get_model_registry method."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        model_registry = config_manager.get_model_registry()
        assert isinstance(model_registry, DictConfig)
        assert "llm_models" in model_registry
        assert "tts_models" in model_registry
        assert "stt_models" in model_registry

    @pytest.mark.unit
    def test_get_vad_params(self, voice_agent_server_base_path):
        """Test get_vad_params method."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        vad_params = config_manager.get_vad_params()

        assert isinstance(vad_params, VADParams)
        assert isinstance(vad_params.confidence, float) and 0.0 <= vad_params.confidence <= 1.0
        assert isinstance(vad_params.start_secs, float) and 0.0 <= vad_params.start_secs <= 1.0
        assert isinstance(vad_params.stop_secs, float) and 0.0 <= vad_params.stop_secs <= 1.0
        assert isinstance(vad_params.min_volume, float) and 0.0 <= vad_params.min_volume <= 1.0

    @pytest.mark.unit
    def test_get_stt_params(self, voice_agent_server_base_path):
        """Test get_stt_params method."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        stt_params = config_manager.get_stt_params()

        assert isinstance(stt_params, NeMoSTTInputParams)
        assert isinstance(stt_params.att_context_size, list)
        assert all(isinstance(item, int) for item in stt_params.att_context_size)
        assert isinstance(stt_params.frame_len_in_secs, float) and 0.0 <= stt_params.frame_len_in_secs <= 1.0
        assert (
            isinstance(stt_params.raw_audio_frame_len_in_secs, float)
            and 0.0 <= stt_params.raw_audio_frame_len_in_secs <= 1.0
        )

    @pytest.mark.unit
    def test_get_diar_params(self, voice_agent_server_base_path):
        """Test get_diar_params method."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        diar_params = config_manager.get_diar_params()

        assert isinstance(diar_params, NeMoDiarInputParams)
        assert hasattr(diar_params, "frame_len_in_secs") and isinstance(diar_params.frame_len_in_secs, float)
        assert hasattr(diar_params, "threshold") and isinstance(diar_params.threshold, float)

    @pytest.mark.unit
    def test_transport_configuration(self, voice_agent_server_base_path):
        """Test transport configuration."""
        config_manager = ConfigManager(voice_agent_server_base_path)
        assert hasattr(config_manager, "TRANSPORT_AUDIO_OUT_10MS_CHUNKS")
        if not isinstance(config_manager.TRANSPORT_AUDIO_OUT_10MS_CHUNKS, int):
            raise ValueError(
                f"TRANSPORT_AUDIO_OUT_10MS_CHUNKS is not an integer: {config_manager.TRANSPORT_AUDIO_OUT_10MS_CHUNKS}"
            )
