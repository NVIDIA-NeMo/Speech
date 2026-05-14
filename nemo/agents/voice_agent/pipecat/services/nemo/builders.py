# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Small composable builders for the pipecat services a voice-agent bot uses.

These are thin wrappers around the existing service constructors so bot scripts
can skip the repeated boilerplate of reading ``ConfigManager`` properties. Each
builder is independent: a bot imports only what it needs. Novel services that
aren't covered here can still be constructed inline.
"""

import copy
from datetime import datetime
from typing import Optional

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.llm_service import LLMService
from pipecat.services.openai import BaseOpenAILLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from nemo.agents.voice_agent.pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import AudioLogger
from nemo.agents.voice_agent.pipecat.services.nemo.diar import NemoDiarService
from nemo.agents.voice_agent.pipecat.services.nemo.llm import get_llm_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.stt import get_stt_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.tts import get_tts_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService
from nemo.agents.voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from nemo.agents.voice_agent.utils import ConfigManager


def build_audio_logger(config_manager: ConfigManager) -> Optional[AudioLogger]:
    """Build an AudioLogger if ``transport.record_audio_data`` is enabled."""
    server_config = config_manager.server_config
    if not server_config.transport.get("record_audio_data", False):
        return None
    log_dir = server_config.transport.get("audio_log_dir", "./audio_logs")
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audio_logger = AudioLogger(log_dir=log_dir, session_id=session_id, enabled=True)
    logger.info(f"AudioLogger initialized for session: {session_id} at {log_dir}")
    return audio_logger


def build_vad_analyzer(config_manager: ConfigManager) -> SileroVADAnalyzer:
    """Build the Silero VAD analyzer at the transport's input sample rate."""
    server_config = config_manager.server_config
    sample_rate = server_config.transport.get("audio_in_sample_rate", config_manager.SAMPLE_RATE)
    return SileroVADAnalyzer(sample_rate=sample_rate, params=config_manager.get_vad_params())


def build_ws_transport(
    config_manager: ConfigManager,
    vad_analyzer: SileroVADAnalyzer | None,
    host: str,
    port: int,
) -> WebsocketServerTransport:
    """Build the no-timeout websocket server transport used by all bots."""
    server_config = config_manager.server_config
    return WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            session_timeout=None,
            audio_in_sample_rate=server_config.transport.get("audio_in_sample_rate", config_manager.SAMPLE_RATE),
            audio_out_sample_rate=server_config.transport.get("audio_out_sample_rate", None),
            can_create_user_frames=server_config.transport.get("can_create_user_frames", False),
            audio_out_10ms_chunks=config_manager.TRANSPORT_AUDIO_OUT_10MS_CHUNKS,
        ),
        host=host,
        port=port,
    )


def build_stt(config_manager: ConfigManager, audio_logger: Optional[AudioLogger] = None) -> STTService:
    """Build the NeMo STT service from config."""
    return get_stt_service_from_config(config_manager.server_config.stt, audio_logger)


def build_diar(config_manager: ConfigManager, audio_logger: Optional[AudioLogger] = None) -> Optional[NemoDiarService]:
    """Build the diarization service, or return ``None`` if ``diar.enabled`` is False."""
    if not config_manager.server_config.diar.get("enabled", False):
        return None
    return NemoDiarService(
        model=config_manager.DIAR_MODEL,
        device=config_manager.STT_DEVICE,
        params=config_manager.get_diar_params(),
        sample_rate=config_manager.SAMPLE_RATE,
        backend="legacy",
        enabled=True,
    )


def build_turn_taking(
    config_manager: ConfigManager,
    audio_logger: Optional[AudioLogger] = None,
    *,
    use_diar: Optional[bool] = None,
    use_vad: bool = True,
) -> NeMoTurnTakingService:
    """Build the turn-taking service. ``use_diar`` defaults to ``config_manager.USE_DIAR``."""
    if use_diar is None:
        use_diar = config_manager.USE_DIAR
    if not config_manager.server_config.turn_taking.get("enabled", True):
        return None
    return NeMoTurnTakingService(
        use_vad=use_vad,
        use_diar=use_diar,
        max_buffer_size=config_manager.TURN_TAKING_MAX_BUFFER_SIZE,
        bot_stop_delay=config_manager.TURN_TAKING_BOT_STOP_DELAY,
        backchannel_phrases=config_manager.TURN_TAKING_BACKCHANNEL_PHRASES_PATH,
        audio_logger=audio_logger,
    )


def build_tts(config_manager: ConfigManager, audio_logger: Optional[AudioLogger] = None) -> TTSService:
    """Build the TTS service via ``get_tts_service_from_config``."""
    return get_tts_service_from_config(config_manager.server_config.tts, audio_logger)


def build_llm(config_manager: ConfigManager) -> LLMService:
    """Build the LLM service via ``get_llm_service_from_config``."""
    return get_llm_service_from_config(config_manager.server_config.llm)


def build_context_and_aggregators(llm: BaseOpenAILLMService, config_manager: ConfigManager):
    """Build ``OpenAILLMContext`` and its user/assistant aggregators.

    Returns ``(context, user_aggregator, assistant_aggregator, original_messages)``.
    ``original_messages`` is a fresh deep-copy of the initial message list, safe
    to hand to the reset/update-prompt RTVI action factories.
    """
    messages = [
        {
            "role": config_manager.SYSTEM_ROLE,
            "content": config_manager.SYSTEM_PROMPT,
        }
    ]
    if config_manager.server_config.llm.get("inject_dummy_user_message", False):
        dummy_message = config_manager.server_config.llm.get("dummy_user_message", "Hello.")
        messages.append({"role": "user", "content": dummy_message})

    context = OpenAILLMContext(messages=messages)
    original_messages = copy.deepcopy(context.get_messages())

    context_aggregator = llm.create_context_aggregator(context)
    return context, context_aggregator.user(), context_aggregator.assistant(), original_messages


def resolve_log_file_path(
    config_manager: ConfigManager, default_name: str = "bot_server.log"
) -> tuple[str, str, bool]:
    """Read the ``server.{log_file,log_level,create_new_log,overwrite_existing_log}`` block.

    Returns ``(log_file, log_level, create_new_log)``. Callers pair this with
    ``setup_rotating_log`` from ``nemo.agents.voice_agent.utils.misc`` to handle
    the rename-existing-log dance.
    """
    server = config_manager.server_config.server
    return (
        server.get("log_file", default_name),
        server.get("log_level", "DEBUG"),
        server.get("create_new_log", False),
    )


def overwrite_existing_log(config_manager: ConfigManager) -> bool:
    """Whether to delete (True) or rename (False) a pre-existing log file on startup."""
    return bool(config_manager.server_config.server.get("overwrite_existing_log", False))


__all__ = [
    "build_audio_logger",
    "build_vad_analyzer",
    "build_ws_transport",
    "build_stt",
    "build_diar",
    "build_turn_taking",
    "build_tts",
    "build_llm",
    "build_context_and_aggregators",
    "resolve_log_file_path",
    "overwrite_existing_log",
]
