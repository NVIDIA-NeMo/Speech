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
import os

from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor

from nemo.agents.voice_agent.evaluation.tools import get_schema_tool_for_eval
from nemo.agents.voice_agent.pipecat.bot_server import (
    create_fastapi_app,
    run_bot_websocket_server,
    run_bot_with_fastapi,
)
from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi import RTVIObserver
from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    SharedStateRef,
    TaskRef,
    create_get_context_history_action,
    create_get_scenario_summary_action,
    create_reset_context_action,
    create_update_system_prompt_action,
)
from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import RTVIAudioLoggerObserver
from nemo.agents.voice_agent.pipecat.services.nemo.builders import (
    build_audio_logger,
    build_context_and_aggregators,
    build_llm,
    build_stt,
    build_tts,
    build_turn_taking,
    build_vad_analyzer,
    build_ws_transport,
)
from nemo.agents.voice_agent.utils import ConfigManager, setup_rotating_log
from nemo.agents.voice_agent.utils.tool_calling import register_schema_tools_to_llm

load_dotenv(override=True)
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8766))
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 7861))
SERVER_CONFIG_PATH = os.getenv("SERVER_CONFIG_PATH", "server_configs/user.yaml")


async def run_bot_websocket(
    server_base_path: str = os.path.dirname(__file__),
    server_config_path: str = "server_configs/user.yaml",
    host: str = "0.0.0.0",
    port: int = 8766,
):
    """Start the evaluation user websocket server; runs until Ctrl+C."""
    logger.info(f"Starting websocket server on {host}:{port} with server config path: {server_config_path}")

    config_manager = ConfigManager(server_base_path=server_base_path, server_config_path=server_config_path)
    server_config = config_manager.get_server_config()
    logger.info(f"Server config: {OmegaConf.to_container(server_config, resolve=True)}")

    log_file = server_config.server.get("log_file", "bot_server.log")
    log_level = server_config.server.get("log_level", "DEBUG")
    setup_rotating_log(
        log_file=log_file,
        log_level=log_level,
        create_new_log=server_config.server.get("create_new_log", False),
        overwrite_existing=server_config.server.get("overwrite_existing_log", False),
    )

    talk_first = server_config.server.get("talk_first", True)
    logger.info(f"Server configured to {'TALK' if talk_first else 'LISTEN'} first")

    audio_logger = build_audio_logger(config_manager)
    vad_analyzer = build_vad_analyzer(config_manager)
    ws_transport = build_ws_transport(config_manager, vad_analyzer, host, port)
    stt = build_stt(config_manager, audio_logger)
    turn_taking = build_turn_taking(config_manager, audio_logger, use_diar=False)
    tts = build_tts(config_manager, audio_logger)

    setup_rotating_log(log_file=log_file, log_level=log_level)

    llm = build_llm(config_manager)
    context, user_agg, assistant_agg, original_messages = build_context_and_aggregators(llm, config_manager)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [ws_transport.input(), rtvi, stt, turn_taking, user_agg, llm, tts, ws_transport.output(), assistant_agg]
    )

    resettable = [stt, tts, turn_taking]
    task_ref = TaskRef()
    shared_state_ref = SharedStateRef()
    rtvi.register_action(create_reset_context_action(task_ref, user_agg, assistant_agg, original_messages, resettable))
    rtvi.register_action(
        create_update_system_prompt_action(
            task_ref,
            user_agg,
            assistant_agg,
            original_messages,
            resettable,
            system_role=config_manager.SYSTEM_ROLE,
            system_prompt_suffix=config_manager.SYSTEM_PROMPT_SUFFIX,
            enable_tool_calling=server_config.llm.get("enable_tool_calling", False),
            llm=llm,
            context=context,
            rtvi=rtvi,
            tool_factory=get_schema_tool_for_eval,
            register_schema_tools=register_schema_tools_to_llm,
            shared_state_ref=shared_state_ref,
        )
    )
    rtvi.register_action(create_get_context_history_action(task_ref, assistant_agg))
    rtvi.register_action(create_get_scenario_summary_action(shared_state_ref))

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,
            enable_usage_metrics=False,
            idle_timeout=None,
        ),
        observers=[RTVIObserver(rtvi), RTVIAudioLoggerObserver(audio_logger=audio_logger)],
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )

    setup_rotating_log(log_file=log_file, log_level=log_level)

    await run_bot_websocket_server(
        task=task,
        ws_transport=ws_transport,
        rtvi=rtvi,
        task_ref=task_ref,
        audio_logger=audio_logger,
        talk_first=talk_first,
        initial_frame_factory=LLMRunFrame,
        on_disconnect_reset_services=resettable,
    )


app = create_fastapi_app(WEBSOCKET_PORT)


async def main():
    logger.info(
        f"Starting servers with config path {SERVER_CONFIG_PATH}, "
        f"WebSocket on port {WEBSOCKET_PORT}, FastAPI on port {FASTAPI_PORT}"
    )
    await run_bot_with_fastapi(
        ws_coro=run_bot_websocket(
            server_config_path=SERVER_CONFIG_PATH,
            host=SERVER_HOST,
            port=WEBSOCKET_PORT,
        ),
        app=app,
        host=SERVER_HOST,
        fastapi_port=FASTAPI_PORT,
    )


if __name__ == "__main__":
    asyncio.run(main())
