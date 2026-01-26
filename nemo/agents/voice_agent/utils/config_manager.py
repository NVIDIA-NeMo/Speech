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
from typing import Any, Dict, Optional

from loguru import logger
from omegaconf import OmegaConf
from pipecat.audio.vad.silero import VADParams

from nemo.agents.voice_agent.pipecat.services.nemo.diar import NeMoDiarInputParams
from nemo.agents.voice_agent.pipecat.services.nemo.stt import NeMoSTTInputParams


class ConfigManager:
    """
    Manages configuration for the voice agent server.
    Handles loading, merging, and providing access to all configuration parameters.
    """

    def __init__(self, server_base_path: str, server_config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the main server configuration file.
                        If None, uses default path from environment variable.
        """
        if not os.path.exists(server_base_path):
            raise FileNotFoundError(f"Server base path not found at {server_base_path}")

        self._server_base_path = server_base_path
        if server_config_path is not None:
            self._server_config_path = server_config_path
        else:
            self._server_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/default.yaml"

        if not os.path.exists(self._server_config_path):
            raise FileNotFoundError(f"Server configuration file not found at {self._server_config_path}")

        # Load model registry
        self.model_registry_path = f"{os.path.abspath(self._server_base_path)}/model_registry.yaml"
        self.model_registry = self._load_model_registry()

        # Load and process main configuration
        self.server_config = self._load_server_config()

        # Initialize configuration parameters
        self._initialize_config_parameters()

        self._generic_hf_llm_model_id = "hf_llm_generic"

        logger.info(f"Configuration loaded from: {self._server_config_path}")
        logger.info(f"Model registry loaded from: {self.model_registry_path}")

    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry from YAML file."""
        try:
            return OmegaConf.load(self.model_registry_path)
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            raise ValueError(f"Failed to load model registry: {e}")

    def _load_server_config(self) -> OmegaConf:
        """Load and process the main server configuration."""
        server_config = OmegaConf.load(self._server_config_path)
        server_config = OmegaConf.to_container(server_config, resolve=True)
        server_config = OmegaConf.create(server_config)
        return server_config

    def _initialize_config_parameters(self):
        """Initialize all configuration parameters from the loaded config."""
        # Default constants
        self.SAMPLE_RATE = 16000
        self.RAW_AUDIO_FRAME_LEN_IN_SECS = 0.016
        self.SYSTEM_PROMPT = " ".join(
            [
                "You are a helpful AI agent named Lisa.",
                "Begin by warmly greeting the user and introducing yourself in one sentence.",
                "Keep your answers concise and to the point.",
            ]
        )

        # Transport configuration
        self.TRANSPORT_AUDIO_OUT_10MS_CHUNKS = self.server_config.transport.audio_out_10ms_chunks

        # VAD configuration
        self.vad_params = VADParams(
            confidence=self.server_config.vad.confidence,
            start_secs=self.server_config.vad.start_secs,
            stop_secs=self.server_config.vad.stop_secs,
            min_volume=self.server_config.vad.min_volume,
        )
        # STT configuration
        self._configure_stt()

        # Diarization configuration
        self._configure_diarization()

        # Turn taking configuration
        self._configure_turn_taking()

        # LLM configuration
        self._configure_llm()

        # TTS configuration
        self._configure_tts()

    def _configure_stt(self):
        """Configure STT parameters."""
        self.STT_MODEL = self.server_config.stt.model
        self.STT_DEVICE = self.server_config.stt.device
        # Apply STT-specific configuration based on model type
        # Try to get STT config file name from server config first
        if self.server_config.stt.get("model_config", None) is not None:
            yaml_file_name = os.path.basename(self.server_config.stt.model_config)
        else:
            # Get STT configuration from registry
            if str(self.STT_MODEL).endswith(".nemo"):
                model_name = os.path.splitext(os.path.basename(self.STT_MODEL))[0]
            else:
                model_name = self.STT_MODEL
            if model_name in self.model_registry.stt_models:
                yaml_file_name = self.model_registry.stt_models[model_name].yaml_id
            else:
                error_msg = f"STT model {model_name} is not in model registry: {self.model_registry.stt_models}."
                logger.error(error_msg)
                raise ValueError(error_msg)

        stt_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/stt_configs/{yaml_file_name}"
        if not os.path.exists(stt_config_path):
            raise FileNotFoundError(f"STT config file not found at {stt_config_path}")
        stt_config = OmegaConf.load(stt_config_path)

        # merge stt config with server config
        for key in stt_config:
            if key in self.server_config.stt and self.server_config.stt[key] != stt_config[key]:
                logger.info(
                    f"STT config field `{key}` is overridden from `{self.server_config.stt[key]}` "
                    f"to `{stt_config[key]}` by {stt_config_path}"
                )
            self.server_config.stt[key] = stt_config[key]

        logger.info(f"Final STT config: {self.server_config.stt}")

        self.stt_params = NeMoSTTInputParams(
            att_context_size=self.server_config.stt.att_context_size,
            frame_len_in_secs=self.server_config.stt.frame_len_in_secs,
            raw_audio_frame_len_in_secs=self.RAW_AUDIO_FRAME_LEN_IN_SECS,
        )

    def _configure_diarization(self):
        """
        Configure diarization parameters.
        Currently only NeMo End-to-End Diarization is supported.
        """
        self.DIAR_MODEL = self.server_config.diar.model
        self.USE_DIAR = self.server_config.diar.enabled
        self.diar_params = NeMoDiarInputParams(
            frame_len_in_secs=self.server_config.diar.frame_len_in_secs,
            threshold=self.server_config.diar.threshold,
        )

    def _configure_turn_taking(self):
        """Configure turn taking parameters."""
        self.TURN_TAKING_BACKCHANNEL_PHRASES_PATH = self.server_config.turn_taking.backchannel_phrases_path
        self.TURN_TAKING_MAX_BUFFER_SIZE = self.server_config.turn_taking.max_buffer_size
        self.TURN_TAKING_BOT_STOP_DELAY = self.server_config.turn_taking.bot_stop_delay

    def _configure_llm(self):
        """Configure LLM parameters."""
        llm_model_id = self.server_config.llm.model
        is_registry_model = False

        # Try to get LLM config file name from server config first
        if self.server_config.llm.get("model_config", None) is not None:
            yaml_file_name = os.path.basename(self.server_config.llm.model_config)
        else:
            # Get LLM configuration from registry
            if llm_model_id in self.model_registry.llm_models:
                yaml_file_name = self.model_registry.llm_models[llm_model_id].yaml_id
                is_registry_model = True
            else:
                logger.warning(
                    f"LLM model {llm_model_id} is not included in the model registry. "
                    "Using a generic HuggingFace LLM config instead."
                )
                yaml_file_name = self.model_registry.llm_models[self._generic_hf_llm_model_id].yaml_id

        # Load and merge LLM configuration
        llm_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/llm_configs/{yaml_file_name}"

        if (
            is_registry_model
            and self.model_registry.llm_models[llm_model_id].get("reasoning_supported", False)
            and self.server_config.llm.get("enable_reasoning", False)
        ):
            llm_config_path = llm_config_path.replace(".yaml", "_think.yaml")

        if not os.path.exists(llm_config_path):
            raise FileNotFoundError(f"LLM config file not found at {llm_config_path}")
        logger.info(f"Loading LLM config from: {llm_config_path}")

        llm_config = OmegaConf.load(llm_config_path)
        # merge llm config with server config
        # print the override keys
        for key in llm_config:
            if key in self.server_config.llm and self.server_config.llm[key] != llm_config[key]:
                logger.info(
                    f"LLM config field `{key}` is overridden from `{self.server_config.llm[key]}` to "
                    f"`{llm_config[key]}` by {llm_config_path}"
                )
            self.server_config.llm[key] = llm_config[key]

        logger.info(f"Final LLM config: {self.server_config.llm}")

        # Configure system prompt
        self.SYSTEM_ROLE = self.server_config.llm.get("system_role", "system")
        if self.server_config.llm.get("system_prompt", None) is not None:
            system_prompt = self.server_config.llm.system_prompt
            if os.path.isfile(system_prompt):
                with open(system_prompt, "r") as f:
                    system_prompt = f.read()
            self.SYSTEM_PROMPT = system_prompt
        else:
            logger.info(f"No system prompt provided, using default system prompt: {self.SYSTEM_PROMPT}")

        if self.server_config.llm.get("system_prompt_suffix", None) is not None:
            self.SYSTEM_PROMPT += "\n" + self.server_config.llm.system_prompt_suffix
            logger.info(f"Adding system prompt suffix: {self.server_config.llm.system_prompt_suffix}")

        logger.info(f"System prompt: {self.SYSTEM_PROMPT}")

    def _configure_tts(self):
        """Configure TTS parameters."""
        tts_model_id = self.server_config.tts.model

        # Try to get TTS config file name from server config first
        if self.server_config.tts.get("model_config", None) is not None:
            yaml_file_name = os.path.basename(self.server_config.tts.model_config)
        else:
            # Get TTS configuration from registry
            if tts_model_id in self.model_registry.tts_models:
                yaml_file_name = self.model_registry.tts_models[tts_model_id].yaml_id
            else:
                error_msg = f"TTS model {tts_model_id} is not in model registry: {self.model_registry.tts_models}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        tts_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/tts_configs/{yaml_file_name}"
        if not os.path.exists(tts_config_path):
            raise FileNotFoundError(f"Default TTS config file not found at {tts_config_path}")
        tts_config = OmegaConf.load(tts_config_path)

        # merge tts config with server config
        for key in tts_config:
            if key in self.server_config.tts and self.server_config.tts[key] != tts_config[key]:
                logger.info(
                    f"TTS config field `{key}` is overridden from `{self.server_config.tts[key]}` to "
                    f"`{tts_config[key]}` by {tts_config_path}"
                )
            self.server_config.tts[key] = tts_config[key]

        logger.info(f"Final TTS config: {self.server_config.tts}")

        # Extract TTS parameters
        self.TTS_MAIN_MODEL_ID = self.server_config.tts.get("main_model_id", None)
        self.TTS_SUB_MODEL_ID = self.server_config.tts.get("sub_model_id", None)
        self.TTS_DEVICE = self.server_config.tts.get("device", None)

        # Handle optional TTS parameters
        self.TTS_THINK_TOKENS = self.server_config.tts.get("think_tokens", None)
        if self.TTS_THINK_TOKENS is not None:
            self.TTS_THINK_TOKENS = OmegaConf.to_container(self.TTS_THINK_TOKENS)

        self.TTS_EXTRA_SEPARATOR = self.server_config.tts.get("extra_separator", None)
        if self.TTS_EXTRA_SEPARATOR is not None:
            self.TTS_EXTRA_SEPARATOR = OmegaConf.to_container(self.TTS_EXTRA_SEPARATOR)

    def get_server_config(self) -> OmegaConf:
        """Get the complete server configuration."""
        return self.server_config

    def get_model_registry(self) -> Dict[str, Any]:
        """Get the model registry configuration."""
        return self.model_registry

    def get_vad_params(self) -> VADParams:
        """Get VAD parameters."""
        return self.vad_params

    def get_stt_params(self) -> NeMoSTTInputParams:
        """Get STT parameters."""
        return self.stt_params

    def get_diar_params(self) -> NeMoDiarInputParams:
        """Get diarization parameters."""
        return self.diar_params
