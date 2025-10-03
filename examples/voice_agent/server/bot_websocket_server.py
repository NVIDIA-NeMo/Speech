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


import asyncio
import copy
import os
import signal
import sys

from loguru import logger
from omegaconf import OmegaConf

# Configure loguru to output to both console and file
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)

logger.add("bot_server.log", rotation="1 day", level="DEBUG")

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIAction, RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from nemo.agents.voice_agent.pipecat.services.nemo.diar import NeMoDiarInputParams, NemoDiarService
from nemo.agents.voice_agent.pipecat.services.nemo.llm import get_llm_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.stt import NeMoSTTInputParams, NemoSTTService
from nemo.agents.voice_agent.pipecat.services.nemo.tts import NeMoFastPitchHiFiGANTTSService
from nemo.agents.voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService
from nemo.agents.voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from nemo.agents.voice_agent.pipecat.utils.text.simple_text_aggregator import SimpleSegmentedTextAggregator
from nemo.agents.voice_agent.utils.config_manager import ConfigManager

# Initialize configuration manager
config_manager = ConfigManager(
    server_base_path=os.path.dirname(__file__), server_config_path=os.environ.get("SERVER_CONFIG_PATH", None)
)
server_config = config_manager.get_server_config()

logger.info(f"Server config: {server_config}")

# Access configuration parameters from ConfigManager
SAMPLE_RATE = config_manager.SAMPLE_RATE
RAW_AUDIO_FRAME_LEN_IN_SECS = config_manager.RAW_AUDIO_FRAME_LEN_IN_SECS
SYSTEM_PROMPT = config_manager.SYSTEM_PROMPT
SYSTEM_ROLE = config_manager.SYSTEM_ROLE

# Transport configuration
TRANSPORT_AUDIO_OUT_10MS_CHUNKS = config_manager.TRANSPORT_AUDIO_OUT_10MS_CHUNKS

# VAD configuration
vad_params = config_manager.get_vad_params()

# STT configuration
STT_MODEL_PATH = config_manager.STT_MODEL_PATH
STT_DEVICE = config_manager.STT_DEVICE
stt_params = config_manager.get_stt_params()

# Diarization configuration
DIAR_MODEL = config_manager.DIAR_MODEL
USE_DIAR = config_manager.USE_DIAR
diar_params = config_manager.get_diar_params()

# Turn taking configuration
TURN_TAKING_BACKCHANNEL_PHRASES_PATH = config_manager.TURN_TAKING_BACKCHANNEL_PHRASES_PATH
TURN_TAKING_MAX_BUFFER_SIZE = config_manager.TURN_TAKING_MAX_BUFFER_SIZE
TURN_TAKING_BOT_STOP_DELAY = config_manager.TURN_TAKING_BOT_STOP_DELAY

# TTS configuration
TTS_MAIN_MODEL_ID = config_manager.TTS_MAIN_MODEL_ID
TTS_SUB_MODEL_ID = config_manager.TTS_SUB_MODEL_ID
TTS_DEVICE = config_manager.TTS_DEVICE
TTS_THINK_TOKENS = config_manager.TTS_THINK_TOKENS
TTS_EXTRA_SEPARATOR = config_manager.TTS_EXTRA_SEPARATOR


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


async def run_bot_websocket_server():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Initializing WebSocket server transport...")
    logger.info("Server configured to run indefinitely with no timeouts")

    """
    NO-TIMEOUT CONFIGURATION:
    - session_timeout=None: Disables WebSocket session timeout
    - idle_timeout=None: Disables pipeline idle timeout  
    - asyncio.wait_for(timeout=None): No timeout on pipeline runner
    - Server will run indefinitely until manually stopped (Ctrl+C)
    """

    vad_analyzer = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=vad_params,
    )
    logger.info("VAD analyzer initialized")

    ws_transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            session_timeout=None,  # Disable session timeout
            audio_in_sample_rate=SAMPLE_RATE,
            can_create_user_frames=TURN_TAKING_BACKCHANNEL_PHRASES_PATH
            is None,  # if backchannel phrases are disabled, we can use VAD to interrupt the bot immediately
            audio_out_10ms_chunks=TRANSPORT_AUDIO_OUT_10MS_CHUNKS,
        ),
        host="0.0.0.0",  # Bind to all interfaces
        port=8765,
    )

    logger.info("Initializing STT service...")

    stt = NemoSTTService(
        model=STT_MODEL_PATH,
        device=STT_DEVICE,
        params=stt_params,
        sample_rate=SAMPLE_RATE,
        audio_passthrough=True,
        has_turn_taking=True,
        backend="legacy",
        decoder_type="rnnt",
    )
    logger.info("STT service initialized")

    if USE_DIAR:
        diar = NemoDiarService(
            model=DIAR_MODEL,
            device=STT_DEVICE,
            params=diar_params,
            sample_rate=SAMPLE_RATE,
            backend="legacy",
            enabled=USE_DIAR,
        )
        logger.info("Diarization service initialized")
    else:
        diar = None

    turn_taking = NeMoTurnTakingService(
        use_vad=True,
        use_diar=USE_DIAR,
        max_buffer_size=TURN_TAKING_MAX_BUFFER_SIZE,
        bot_stop_delay=TURN_TAKING_BOT_STOP_DELAY,
        backchannel_phrases=TURN_TAKING_BACKCHANNEL_PHRASES_PATH,
    )
    logger.info("Turn taking service initialized")

    logger.info("Initializing LLM service...")

    llm = get_llm_service_from_config(server_config.llm)
    logger.info("LLM service initialized")

    text_aggregator = SimpleSegmentedTextAggregator(punctuation_marks=TTS_EXTRA_SEPARATOR)

    tts = NeMoFastPitchHiFiGANTTSService(
        fastpitch_model=TTS_MAIN_MODEL_ID,
        hifigan_model=TTS_SUB_MODEL_ID,
        device=TTS_DEVICE,
        text_aggregator=text_aggregator,
        think_tokens=TTS_THINK_TOKENS,
    )

    logger.info("TTS service initialized")

    context = OpenAILLMContext(
        [
            {
                "role": SYSTEM_ROLE,
                "content": SYSTEM_PROMPT,
            }
        ],
    )

    original_messages = copy.deepcopy(context.get_messages())
    original_context = copy.deepcopy(context)
    original_context.set_llm_adapter(llm.get_llm_adapter())

    context_aggregator = llm.create_context_aggregator(context)
    user_context_aggregator = context_aggregator.user()
    assistant_context_aggregator = context_aggregator.assistant()

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Add reset action to RTVI processor
    async def reset_context_handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, any]) -> bool:
        """Reset both user and assistant context aggregators"""
        logger.info("Resetting conversation context...")
        try:
            user_context_aggregator.reset()
            assistant_context_aggregator.reset()
            user_context_aggregator.set_messages(copy.deepcopy(original_messages))
            assistant_context_aggregator.set_messages(copy.deepcopy(original_messages))

            logger.info("Conversation context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting context: {e}")
            return False

    reset_action = RTVIAction(
        service="context",
        action="reset",
        result="bool",
        arguments=[],
        handler=reset_context_handler,
    )
    rtvi.register_action(reset_action)

    logger.info("Setting up pipeline...")

    pipeline = [
        ws_transport.input(),
        rtvi,
        stt,
    ]

    if USE_DIAR:
        pipeline.append(diar)

    pipeline.extend(
        [turn_taking, user_context_aggregator, llm, tts, ws_transport.output(), assistant_context_aggregator]
    )

    pipeline = Pipeline(pipeline)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
            enable_usage_metrics=False,
            send_initial_empty_metrics=True,
            report_only_initial_ttfb=True,
            idle_timeout=None,  # Disable idle timeout
        ),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )

    # Track task state
    task_running = True

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi: RTVIProcessor):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        # Kick off the conversation.
        try:
            await task.queue_frames([user_context_aggregator.get_context_frame()])
        except Exception as e:
            logger.error(f"Error queuing context frame: {e}")

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Pipecat Client connected from {client.remote_address}")
        # Reset RTVI state for new connection
        rtvi._client_ready = False
        rtvi._bot_ready = False

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Pipecat Client disconnected from {client.remote_address}")
        # Don't cancel the task immediately - let it handle the disconnection gracefully
        # The task will continue running and can accept new connections
        # Only send an EndTaskFrame to clean up the current session
        if task_running:
            try:
                await task.queue_frames([EndTaskFrame()])
            except Exception as e:
                # Don't log warnings for normal connection closures
                if "ConnectionClosedOK" not in str(e) and "1005" not in str(e):
                    logger.warning(f"Error sending EndTaskFrame: {e}")
                else:
                    logger.debug(f"Normal connection closure: {e}")

    @ws_transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Session timeout for {client.remote_address}")
        # Don't cancel the task - keep server running indefinitely
        logger.info("Session timeout occurred but keeping server running")
        # Note: With session_timeout=None, this handler should never be called

    logger.info("Starting pipeline runner...")

    try:
        runner = PipelineRunner()
        # Run the task until shutdown is requested
        await asyncio.wait_for(runner.run(task), timeout=None)  # No timeout - run indefinitely
    except asyncio.TimeoutError:
        logger.info("Pipeline runner timeout (should not happen with no timeout)")
    except Exception as e:
        logger.error(f"Pipeline runner error: {e}")
        task_running = False
    finally:
        logger.info("Pipeline runner stopped")


if __name__ == "__main__":
    asyncio.run(run_bot_websocket_server())
