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
import os

from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserverParams, RTVIProcessor

from nemo.agents.voice_agent.pipecat.bot_server import (
    create_fastapi_app,
    run_bot_websocket_server,
    run_bot_with_fastapi,
)
from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi import RTVIObserver
from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi_actions import TaskRef, create_reset_context_action
from nemo.agents.voice_agent.pipecat.services.common import UserAudioBuffer
from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import RTVIAudioLoggerObserver
from nemo.agents.voice_agent.pipecat.services.nemo.builders import (
    build_audio_logger,
    build_context_and_aggregators,
    build_diar,
    build_llm,
    build_stt,
    build_tts,
    build_turn_taking,
    build_vad_analyzer,
    build_ws_transport,
)
from nemo.agents.voice_agent.utils import ConfigManager, setup_logging
from nemo.agents.voice_agent.utils.tool_calling.basic_tools import tool_get_city_weather
from nemo.agents.voice_agent.utils.tool_calling.mixins import register_direct_tools_to_llm

load_dotenv(override=True)

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8765))
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 7860))
SERVER_CONFIG_PATH = os.getenv("SERVER_CONFIG_PATH", None)


async def run_bot_websocket(host: str, port: int):
    """Start the production bot websocket server; runs until Ctrl+C."""
    logger.info(f"Starting websocket server on {host}:{port}")
    logger.info("Server configured to run indefinitely with no timeouts, use Ctrl+C to quit.")

    setup_logging()

    config_manager = ConfigManager(server_base_path=os.path.dirname(__file__), server_config_path=SERVER_CONFIG_PATH)
    server_config = config_manager.get_server_config()
    logger.info(f"Server config: {OmegaConf.to_container(server_config, resolve=True)}")

    audio_logger = build_audio_logger(config_manager)
    vad_analyzer = build_vad_analyzer(config_manager)
    ws_transport = build_ws_transport(config_manager, vad_analyzer, host, port)
    stt = build_stt(config_manager, audio_logger)
    diar = build_diar(config_manager, audio_logger)
    turn_taking = build_turn_taking(config_manager, audio_logger)
    tts = build_tts(config_manager, audio_logger)

    setup_logging()

    llm = build_llm(config_manager)
    context, user_agg, assistant_agg, original_messages = build_context_and_aggregators(llm, config_manager)

    if server_config.llm.get("is_omni_model", False):
        user_audio_buffer = UserAudioBuffer(
            context=context,
            user_context_aggregator=user_agg,
            pre_cache_duration_secs=server_config.llm.get("pre_cache_duration_secs", 0.3),
            use_transcript=server_config.llm.get("use_stt_transcript", False),
        )
    else:
        user_audio_buffer = None

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    pipeline_list = [ws_transport.input(), rtvi, stt]
    if diar is not None:
        pipeline_list.append(diar)
    if user_audio_buffer is not None:
        pipeline_list.append(user_audio_buffer)
    pipeline_list.extend([turn_taking, user_agg, llm, tts, ws_transport.output(), assistant_agg])
    pipeline = Pipeline(pipeline_list)

    resettable = [stt, tts, turn_taking, diar, user_audio_buffer]

    if server_config.llm.get("enable_tool_calling", False):
        logger.info("Tool calling enabled; registering initial tools...")
        register_direct_tools_to_llm(llm=llm, context=context, tool_mixins=[tts], tools=[tool_get_city_weather])
    else:
        logger.info("Tool calling disabled; skipping initial tool registration.")

    task_ref = TaskRef()
    rtvi.register_action(create_reset_context_action(task_ref, user_agg, assistant_agg, original_messages, resettable))

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
            enable_usage_metrics=False,
            send_initial_empty_metrics=True,
            report_only_initial_ttfb=True,
            idle_timeout=None,
        ),
        observers=[
            RTVIObserver(rtvi, params=RTVIObserverParams()),
            RTVIAudioLoggerObserver(audio_logger=audio_logger),
        ],
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )

    setup_logging()

    await run_bot_websocket_server(
        task=task,
        ws_transport=ws_transport,
        rtvi=rtvi,
        task_ref=task_ref,
        audio_logger=audio_logger,
        talk_first=True,
        initial_frame_factory=LLMRunFrame,
        on_disconnect_reset_services=None,
    )


app = create_fastapi_app(WEBSOCKET_PORT)


async def main():
    logger.info(f"Starting servers - WebSocket on port {WEBSOCKET_PORT}, FastAPI on port {FASTAPI_PORT}")
    await run_bot_with_fastapi(
        ws_coro=run_bot_websocket(host=SERVER_HOST, port=WEBSOCKET_PORT),
        app=app,
        host=SERVER_HOST,
        fastapi_port=FASTAPI_PORT,
    )


if __name__ == "__main__":
    asyncio.run(main())
