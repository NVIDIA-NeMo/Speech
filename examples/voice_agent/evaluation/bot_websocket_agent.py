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


import asyncio
import copy
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIAction, RTVIConfig, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from nemo.agents.voice_agent.evaluation.tools import get_schema_tool_for_eval
from nemo.agents.voice_agent.evaluation.tools.basic_tools import GetCityWeatherTool
from nemo.agents.voice_agent.evaluation.tools.rtvi_control import SendScenarioSummaryTool
from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi import RTVIObserver
from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import AudioLogger, RTVIAudioLoggerObserver
from nemo.agents.voice_agent.pipecat.services.nemo.diar import NemoDiarService
from nemo.agents.voice_agent.pipecat.services.nemo.llm import get_llm_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.stt import ASR_EOU_MODELS, NemoSTTService
from nemo.agents.voice_agent.pipecat.services.nemo.tts import get_tts_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService
from nemo.agents.voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from nemo.agents.voice_agent.utils import ConfigManager, setup_logging
from nemo.agents.voice_agent.utils.tool_calling import register_schema_tools_to_llm


async def run_bot_websocket_server(
    server_base_path: str = os.path.dirname(__file__),
    server_config_path: str = "server_configs/agent.yaml",
    host: str = "0.0.0.0",
    port: int = 8765,
):
    """
    Creates a websocket server that runs indefinitely until manually stopped (Ctrl+C)
    Args:
        server_config_path: Path to the server configuration file, defaults to `server_configs/default.yaml`
        host: Host to bind the server to, defaults to `0.0.0.0`
        port: Port to bind the server to, defaults to `8765`
    """
    logger.info(f"Starting websocket server on {host}:{port} with server config path: {server_config_path}")

    config_manager = ConfigManager(
        server_base_path=server_base_path,
        server_config_path=server_config_path,
    )
    server_config = config_manager.get_server_config()
    logger.info(f"Server config: {OmegaConf.to_container(server_config, resolve=True)}")
    log_file = server_config.server.get("log_file", "bot_server.log")
    log_level = server_config.server.get("log_level", "DEBUG")
    create_new_log = server_config.server.get("create_new_log", False)

    if create_new_log:
        if os.path.exists(log_file):
            if server_config.server.get("overwrite_existing_log", False):
                os.remove(log_file)
                logger.info(f"Removed existing log file: {log_file}")
            else:
                # Rename the existing log file to the current timestamp
                new_log_file = log_file.replace(".log", f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                os.rename(log_file, new_log_file)
                logger.info(f"Renamed existing log file: {log_file} to {new_log_file}")

    setup_logging(log_file=log_file, log_level=log_level)

    # Access configuration parameters from ConfigManager
    SAMPLE_RATE = config_manager.SAMPLE_RATE
    SYSTEM_PROMPT = config_manager.SYSTEM_PROMPT
    SYSTEM_PROMPT_SUFFIX = config_manager.SYSTEM_PROMPT_SUFFIX
    SYSTEM_ROLE = config_manager.SYSTEM_ROLE
    TALK_FIRST = server_config.server.get("talk_first", True)
    if TALK_FIRST:
        logger.info("Server configured to TALK first")
    else:
        logger.info("Server configured to LISTEN first")

    # Transport configuration
    TRANSPORT_AUDIO_IN_SAMPLE_RATE = server_config.transport.get("audio_in_sample_rate", SAMPLE_RATE)
    TRANSPORT_AUDIO_OUT_10MS_CHUNKS = config_manager.TRANSPORT_AUDIO_OUT_10MS_CHUNKS
    TRANSPORT_AUDIO_OUT_SAMPLE_RATE = server_config.transport.get(
        "audio_out_sample_rate", None
    )  # None means use pipecat default
    RECORD_AUDIO_DATA = server_config.transport.get("record_audio_data", False)
    AUDIO_LOG_DIR = server_config.transport.get("audio_log_dir", "./audio_logs")

    # VAD configuration
    vad_params = config_manager.get_vad_params()

    # STT configuration
    STT_MODEL = config_manager.STT_MODEL
    STT_DEVICE = config_manager.STT_DEVICE
    stt_params = config_manager.get_stt_params()
    ignore_eou_eob = server_config.stt.get("ignore_eou_eob", False)

    # Diarization configuration
    DIAR_MODEL = config_manager.DIAR_MODEL
    USE_DIAR = config_manager.USE_DIAR
    diar_params = config_manager.get_diar_params()

    # Turn taking configuration
    TURN_TAKING_BACKCHANNEL_PHRASES_PATH = config_manager.TURN_TAKING_BACKCHANNEL_PHRASES_PATH
    TURN_TAKING_MAX_BUFFER_SIZE = config_manager.TURN_TAKING_MAX_BUFFER_SIZE
    TURN_TAKING_BOT_STOP_DELAY = config_manager.TURN_TAKING_BOT_STOP_DELAY

    # TTS configuration
    TTS_TYPE = config_manager.server_config.tts.type

    logger.info("Initializing WebSocket server transport...")
    logger.info("Server configured to run indefinitely with no timeouts")
    # Initialize AudioLogger if recording is enabled
    audio_logger = None
    if RECORD_AUDIO_DATA:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        audio_logger = AudioLogger(
            log_dir=AUDIO_LOG_DIR,
            session_id=session_id,
            enabled=True,
        )
        logger.info(f"AudioLogger initialized for session: {session_id} at {AUDIO_LOG_DIR}")

    vad_analyzer = SileroVADAnalyzer(
        sample_rate=TRANSPORT_AUDIO_IN_SAMPLE_RATE,
        params=vad_params,
    )
    logger.info("VAD analyzer initialized")

    has_turn_taking = True if STT_MODEL in ASR_EOU_MODELS else False
    logger.info(f"Setting STT service has_turn_taking to `{has_turn_taking}` based on model name: `{STT_MODEL}`")

    ws_transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            session_timeout=None,  # Disable session timeout
            audio_in_sample_rate=TRANSPORT_AUDIO_IN_SAMPLE_RATE,
            audio_out_sample_rate=TRANSPORT_AUDIO_OUT_SAMPLE_RATE,
            can_create_user_frames=TURN_TAKING_BACKCHANNEL_PHRASES_PATH is None
            or not has_turn_taking,  # if backchannel phrases are disabled, we can use VAD to interrupt the bot immediately
            audio_out_10ms_chunks=TRANSPORT_AUDIO_OUT_10MS_CHUNKS,
        ),
        host=host,
        port=port,
    )

    input_transport = ws_transport.input()
    output_transport = ws_transport.output()

    logger.info("Initializing STT service...")

    stt = NemoSTTService(
        model=STT_MODEL,
        device=STT_DEVICE,
        params=stt_params,
        sample_rate=SAMPLE_RATE,
        audio_passthrough=True,
        has_turn_taking=has_turn_taking,
        backend="legacy",
        decoder_type="rnnt",
        audio_logger=audio_logger,
        ignore_eou_eob=ignore_eou_eob,
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
        audio_logger=audio_logger,
    )
    logger.info("Turn taking service initialized")

    if TTS_TYPE == "nemo":
        tts = get_tts_service_from_config(config_manager.server_config.tts, audio_logger)
    else:
        raise ValueError(f"Invalid TTS type: {TTS_TYPE}")

    logger.info("TTS service initialized")

    # Setup logging again to avoid logger from being overwritten during setting up the pipeline components
    setup_logging(log_file=log_file, log_level=log_level)

    # Put LLM in the end of model initialization to reduce the chance of running out of HBM memory
    logger.info("Initializing LLM service...")
    llm = get_llm_service_from_config(server_config.llm)
    logger.info("LLM service initialized")

    messages = [
        {
            "role": SYSTEM_ROLE,
            "content": SYSTEM_PROMPT,
        }
    ]
    inject_dummy_user_message = server_config.llm.get("inject_dummy_user_message", False)
    if inject_dummy_user_message:
        messages.append(
            {
                "role": "user",
                "content": "Hello, who are you?",
            }
        )
    context = OpenAILLMContext(messages=messages)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    send_final_response_tool = SendScenarioSummaryTool(rtvi=rtvi)
    get_city_weather_tool = GetCityWeatherTool()

    if server_config.llm.get("enable_tool_calling", False):
        logger.info("Tools calling for LLM is enabled by config, registering tools...")
        # register_direct_tools_to_llm(llm=llm, context=context, tool_mixins=[tts])
        register_schema_tools_to_llm(
            llm=llm,
            context=context,
            tools=[send_final_response_tool, get_city_weather_tool],
            cancel_on_interruption=False,
        )
    else:
        logger.info("Tools calling for LLM is disabled by config, skipping tool registration.")

    original_messages = copy.deepcopy(context.get_messages())
    original_context = copy.deepcopy(context)
    original_context.set_llm_adapter(llm.get_llm_adapter())

    context_aggregator = llm.create_context_aggregator(context)
    user_context_aggregator = context_aggregator.user()
    assistant_context_aggregator = context_aggregator.assistant()

    # Add reset action to RTVI processor
    async def reset_context_handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, any]) -> bool:
        """Reset both user and assistant context aggregators"""
        logger.info("Resetting conversation context...")
        try:
            if task_running:
                await task.queue_frames([EndTaskFrame()])
            user_context_aggregator.reset()
            assistant_context_aggregator.reset()
            user_context_aggregator.set_messages(copy.deepcopy(original_messages))
            assistant_context_aggregator.set_messages(copy.deepcopy(original_messages))
            stt.reset()
            tts.reset()
            if turn_taking is not None:
                turn_taking.reset()
            if diar is not None:
                diar.reset()
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

    # Add update_system_prompt action for dynamic prompt updates
    async def update_system_prompt_handler(
        rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, any]
    ) -> bool:
        """Update the system prompt dynamically and reset the conversation"""
        try:
            if task_running:
                await task.queue_frames([EndTaskFrame()])

            # save previous log file by renaming it to a new file with the current timestamp if it exists
            # so that new logs will be written to a new file.
            if os.path.exists(log_file):
                new_log_file = log_file.replace(".log", f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                os.rename(log_file, new_log_file)
                logger.info(f"Renamed existing log file: {log_file} to {new_log_file}")
            setup_logging(log_file=log_file, log_level=log_level)

            new_prompt = arguments.get("prompt", "")
            new_tools = arguments.get("tools", "{}")
            if not new_prompt:
                logger.error("No prompt provided in update_system_prompt action")
                return False

            logger.info(f"Updating system prompt to: {new_prompt[:100]}...")

            add_suffix = arguments.get("add_suffix", True)
            if add_suffix and SYSTEM_PROMPT_SUFFIX:
                new_prompt = f"{new_prompt}\n{SYSTEM_PROMPT_SUFFIX}"

            # Create new messages with updated system prompt
            new_messages = [
                {
                    "role": SYSTEM_ROLE,
                    "content": new_prompt,
                }
            ]

            # Store the new messages in original_messages so reset will use them
            original_messages.clear()
            original_messages.extend(new_messages)

            # Reset the context (this will apply the new prompt)
            user_context_aggregator.reset()
            assistant_context_aggregator.reset()
            user_context_aggregator.set_messages(copy.deepcopy(new_messages))
            assistant_context_aggregator.set_messages(copy.deepcopy(new_messages))

            if server_config.llm.get("enable_tool_calling", False) and new_tools:
                logger.info("Registering new tools...")
                new_tools = json.loads(new_tools)
                new_schema_tools = []
                for tool_name, tool_args in new_tools.items():
                    new_schema_tools.append(get_schema_tool_for_eval(tool_name, rtvi=rtvi, **tool_args))
                register_schema_tools_to_llm(
                    llm=llm,
                    context=context,
                    tools=new_schema_tools,
                    cancel_on_interruption=False,
                    keep_existing_tools=False,  # Override existing tools
                )
            else:
                logger.info(
                    "Tools calling for LLM is disabled by config, or no new tools provided, skipping tool registration..."
                )

            logger.debug(f"user context tools: {user_context_aggregator._context.tools}")
            logger.debug(f"assistant context tools: {assistant_context_aggregator._context.tools}")

            # Reset services
            stt.reset()
            tts.reset()
            if turn_taking is not None:
                turn_taking.reset()
            if diar is not None:
                diar.reset()

            logger.info("System prompt updated and context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating system prompt: {e}")
            return False

    update_prompt_action = RTVIAction(
        service="context",
        action="update_system_prompt",
        result="bool",
        arguments=[
            {
                "name": "prompt",
                "type": "string",
                "required": True,
            },
            {
                "name": "tools",
                "type": "string",
                "required": False,
                "default": "{}",
            },
            {
                "name": "add_suffix",
                "type": "bool",
                "required": False,
                "default": True,
            },
        ],
        handler=update_system_prompt_handler,
    )
    rtvi.register_action(update_prompt_action)

    # Add get_context_history action for retrieving LLM context and server logs
    async def get_context_history_handler(
        rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, any]
    ) -> dict:
        """Return the current LLM context history and server log contents."""
        if task_running:
            await task.queue_frames([EndTaskFrame()])
        try:
            messages = assistant_context_aggregator._context.get_messages()
            log_content = ""
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_content = f.read()
            logger.debug(f"Returning context history: {len(messages)} messages, {len(log_content)} log characters")
            return {"context": str(messages), "logs": log_content}
        except Exception as e:
            logger.error(f"Error getting context history: {e}")
            return {"context": [], "logs": ""}

    get_context_history_action = RTVIAction(
        service="context",
        action="get_context_history",
        result="object",
        arguments=[],
        handler=get_context_history_handler,
    )
    rtvi.register_action(get_context_history_action)

    logger.info("Setting up pipeline...")

    pipeline = [
        input_transport,
        rtvi,
        stt,
    ]

    if USE_DIAR:
        pipeline.append(diar)

    pipeline.extend([turn_taking, user_context_aggregator, llm, tts, output_transport, assistant_context_aggregator])

    pipeline = Pipeline(pipeline)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=server_config.transport.get("allow_interruption", True),
            enable_metrics=False,
            enable_usage_metrics=False,
            send_initial_empty_metrics=True,
            report_only_initial_ttfb=True,
            idle_timeout=None,  # Disable idle timeout
        ),
        observers=[
            RTVIObserver(rtvi),
            RTVIAudioLoggerObserver(audio_logger=audio_logger),
        ],
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )

    # Track task state
    task_running = True

    # Setup logging again to avoid logger from being overwritten during setting up the pipeline components
    setup_logging(log_file=log_file, log_level=log_level)

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi: RTVIProcessor):
        logger.info(f"Pipecat client ready with TALK_FIRST set to {TALK_FIRST}.")
        await rtvi.set_bot_ready()
        if TALK_FIRST:
            # Kick off the conversation.
            try:
                logger.info("Kicking off the conversation...")
                await task.queue_frames([LLMRunFrame()])
            except Exception as e:
                logger.error(f"Error queuing context frame: {e}")
        else:
            logger.info("Pipecat client ready, listening...")

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Pipecat Client connected from {client.remote_address}")
        # Reset RTVI state for new connection
        rtvi._client_ready = False
        rtvi._bot_ready = False

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Pipecat Client disconnected from {client.remote_address}")
        # Finalize audio logger session if enabled
        if audio_logger:
            audio_logger.finalize_session()
            logger.info("Audio logger session finalized")
        # Don't cancel the task immediately - let it handle the disconnection gracefully
        # The task will continue running and can accept new connections
        # Only send an EndTaskFrame to clean up the current session
        if task_running:
            try:
                await task.queue_frames([EndTaskFrame()])
                stt.reset()
                tts.reset()
                if turn_taking is not None:
                    turn_taking.reset()
                if diar is not None:
                    diar.reset()
            except Exception as e:
                # Don't log warnings for normal connection closures
                if "ConnectionClosedOK" not in str(e) and "1005" not in str(e):
                    logger.warning(f"Error sending EndTaskFrame: {e}")
                else:
                    logger.info(f"Normal connection closure: {e}")

    @ws_transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Session timeout for {client.remote_address}")
        # Don't cancel the task - keep server running indefinitely
        logger.info("Session timeout occurred but keeping server running")
        # Note: With session_timeout=None, this handler should never be called
        if audio_logger:
            audio_logger.finalize_session()
            logger.info("Audio logger session finalized")
        if task_running:
            try:
                await task.queue_frames([EndTaskFrame()])
            except Exception as e:
                logger.error(f"Error sending EndTaskFrame: {e}")

    logger.info("Starting pipeline runner...")

    try:
        runner = PipelineRunner()
        # Run the task until shutdown is requested
        await asyncio.wait_for(runner.run(task), timeout=None)  # No timeout - run indefinitely
    except asyncio.TimeoutError:
        task_running = False
        logger.info("Pipeline runner timeout (should not happen with no timeout)")
    except Exception as e:
        logger.error(f"Pipeline runner error: {e}")
    finally:
        # Finalize audio logger on shutdown
        if audio_logger:
            audio_logger.finalize_session()
            logger.info("Audio logger session finalized on shutdown")
        logger.info("Pipeline runner stopped")
        task_running = False


if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(
        run_bot_websocket_server(
            server_config_path=os.getenv("SERVER_CONFIG_PATH", "server_configs/agent.yaml"),
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("WEBSOCKET_PORT", 8765)),
        )
    )
