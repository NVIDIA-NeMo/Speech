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
from enum import Enum
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPSpanExporterGRPC
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.user_bot_latency_log_observer import UserBotLatencyLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.transports.websocket.server import WebsocketServerParams, WebsocketServerTransport
from pipecat.utils.tracing.setup import setup_tracing

from nemo.agents.voice_agent.evaluation.tools import get_schema_tool_for_eval
from nemo.agents.voice_agent.evaluation.tools.basic_tools import GetCityWeatherTool
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
from nemo.agents.voice_agent.pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nemo.agents.voice_agent.pipecat.services.riva_speech import NemotronASRService, NemotronTTSService
from nemo.agents.voice_agent.pipecat.utils.riva_text_filter import RivaTextFilter
from nemo.agents.voice_agent.utils import setup_rotating_log
from nemo.agents.voice_agent.utils.tool_calling import register_schema_tools_to_llm


class VADProfile(Enum):
    """VAD Profile options."""

    SILERO = "Silero"  # Transport Silero VAD analyzer
    ASR = "ASR"  # ASR VAD


load_dotenv(override=True)
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8766))
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 7861))
TALK_FIRST = os.getenv("TALK_FIRST", "true").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "bot_user_nemotron.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")


AUDIO_OUT_10MS_CHUNKS = int(os.getenv("AUDIO_OUT_10MS_CHUNKS", "10"))
ENABLE_MULTILINGUAL = os.getenv("ENABLE_MULTILINGUAL", "false").lower() == "true"
# user VAD with longer stop time to prevent user from interrupting the agent
VAD_PROFILE = VADProfile(os.getenv("VAD_PROFILE", VADProfile.SILERO))
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", 1.2))
PROMPT_FILE = Path(
    os.getenv("PROMPT_FILE_PATH", str(Path(__file__).parent / "nemotron_voice_agent_config" / "prompt.yaml"))
)
IPA_FILE = Path(os.getenv("IPA_FILE_PATH", str(Path(__file__).parent / "nemotron_voice_agent_config" / "ipa.json")))
IS_TRACING_ENABLED = os.getenv("ENABLE_TRACING", "false").lower() == "true"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_LLM_URL = os.getenv("NVIDIA_LLM_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
ENABLE_TOOL_CALLING = os.getenv("ENABLE_TOOL_CALLING", "false").lower() == "true"
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "1500"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
TOP_P = float(os.getenv("TOP_P", "1.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

ASR_SERVER_URL = os.getenv("ASR_SERVER_URL", "grpc.nvcf.nvidia.com:443")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en-US")
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer")
ASR_CLOUD_FUNCTION_ID = os.getenv("ASR_CLOUD_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081")

ENABLE_TTS_TEXT_FILTER = os.getenv("ENABLE_TTS_TEXT_FILTER", "true").lower() == "true"
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "grpc.nvcf.nvidia.com:443")
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Leo")  # default to Aria for agent, Leo for user
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "magpie_tts_ensemble-Magpie-Multilingual")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en-US")
ZERO_SHOT_AUDIO_PROMPT = os.getenv("ZERO_SHOT_AUDIO_PROMPT")

SYSTEM_PROMPT_SELECTOR = os.getenv("SYSTEM_PROMPT_SELECTOR")

ENABLE_SPECULATIVE_SPEECH = os.getenv("ENABLE_SPECULATIVE_SPEECH", "true").lower() == "true"
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", -1))


def _load_prompts() -> dict:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt catalog not found at {PROMPT_FILE}")
    try:
        data = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in prompt catalog {PROMPT_FILE}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Prompt catalog at {PROMPT_FILE} must be a mapping.")
    return data


PROMPTS = _load_prompts()


def _resolve_prompt(selector: str) -> list[dict[str, str]]:
    """Resolve a selector like 'model/prompt' into a list of {role, content} messages."""
    try:
        entry = PROMPTS
        for part in selector.split("/"):
            entry = entry[part]
        return [{"role": m["role"], "content": m["content"]} for m in entry["messages"]]
    except (KeyError, TypeError) as e:
        raise KeyError(f"Prompt '{selector}' not found or invalid: {e}") from e


def _inject_prompt_variables(prompt: str, **variables) -> str:
    """Inject variables into prompt placeholders like {lang_codes}."""
    try:
        return prompt.format(**variables)
    except KeyError:
        return prompt


# Initialize tracing if enabled
if IS_TRACING_ENABLED:
    # Get the endpoint URL
    endpoint_url = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # Determine which exporter to use based on the endpoint URL
    if endpoint_url.startswith("http://") or endpoint_url.startswith("https://"):
        # HTTP exporter - use full URL with protocol
        otlp_exporter = OTLPSpanExporterHTTP(endpoint=endpoint_url)
    else:
        # gRPC exporter - endpoint should be host:port format (no protocol prefix)
        otlp_exporter = OTLPSpanExporterGRPC(endpoint=endpoint_url, insecure=True)

    # Set up tracing with the exporter
    setup_tracing(
        service_name="nemotron-voice-agent",
        exporter=otlp_exporter,
        console_export=os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true",
    )
    logger.info("OpenTelemetry tracing initialized")


async def run_bot_websocket(
    host: str = "0.0.0.0",
    port: int = 8765,
):
    """Start the evaluation agent websocket server; runs until Ctrl+C."""
    logger.info(f"Starting websocket server on {host}:{port}")
    logger.info(f"------- LLM -------")
    logger.info(f"NVIDIA_LLM_URL: {NVIDIA_LLM_URL}")
    logger.info(f"NVIDIA_LLM_MODEL: {NVIDIA_LLM_MODEL}")
    logger.info(f"ENABLE_TOOL_CALLING: {ENABLE_TOOL_CALLING}")
    logger.info(f"ENABLE_THINKING: {ENABLE_THINKING}")
    logger.info(f"THINKING_BUDGET: {THINKING_BUDGET}")
    logger.info(f"------- ASR -------")
    logger.info(f"ASR_SERVER_URL: {ASR_SERVER_URL}")
    logger.info(f"ASR_MODEL_NAME: {ASR_MODEL_NAME}")
    logger.info(f"ASR_LANGUAGE: {ASR_LANGUAGE}")
    logger.info(f"ASR_CLOUD_FUNCTION_ID: {ASR_CLOUD_FUNCTION_ID}")
    logger.info(f"------- TTS -------")
    logger.info(f"TTS_SERVER_URL: {TTS_SERVER_URL}")
    logger.info(f"TTS_VOICE_ID: {TTS_VOICE_ID}")
    logger.info(f"TTS_MODEL_NAME: {TTS_MODEL_NAME}")
    logger.info(f"TTS_LANGUAGE: {TTS_LANGUAGE}")
    logger.info(f"------- Misc ------")
    logger.info(f"ENABLE_MULTILINGUAL: {ENABLE_MULTILINGUAL}")
    logger.info(f"VAD_PROFILE: {VAD_PROFILE}")
    logger.info(f"VAD_STOP_SECS: {VAD_STOP_SECS}")
    logger.info(f"PROMPT_FILE: {PROMPT_FILE}")
    logger.info(f"IPA_FILE: {IPA_FILE}")
    logger.info(f"IS_TRACING_ENABLED: {IS_TRACING_ENABLED}")
    logger.info(f"ZERO_SHOT_AUDIO_PROMPT: {ZERO_SHOT_AUDIO_PROMPT}")
    logger.info(f"SYSTEM_PROMPT_SELECTOR: {SYSTEM_PROMPT_SELECTOR}")
    logger.info(f"ENABLE_SPECULATIVE_SPEECH: {ENABLE_SPECULATIVE_SPEECH}")
    logger.info(f"CHAT_HISTORY_LIMIT: {CHAT_HISTORY_LIMIT}")
    logger.info(f"LOG_FILE: {LOG_FILE}")
    logger.info(f"-------------------")

    setup_rotating_log(
        log_file=LOG_FILE,
        log_level=LOG_LEVEL,
        create_new_log=False,
        overwrite_existing=False,
    )

    talk_first = TALK_FIRST
    logger.info(f"Server configured to {'TALK' if talk_first else 'LISTEN'} first")

    if VAD_PROFILE == VADProfile.SILERO:
        vad_analyzer = SileroVADAnalyzer(
            params=VADParams(
                stop_secs=VAD_STOP_SECS,
            )
        )
    else:
        vad_analyzer = None

    ws_transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            session_timeout=None,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,  # the browser app expects 24kHz
            audio_out_10ms_chunks=AUDIO_OUT_10MS_CHUNKS,
            vad_analyzer=vad_analyzer,
        ),
        host=host,
        port=port,
    )

    enable_thinking = bool(ENABLE_THINKING) if ENABLE_THINKING is not None else False
    thinking_budget = int(THINKING_BUDGET) if THINKING_BUDGET is not None else -1
    if thinking_budget < 0:
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    else:
        if thinking_budget >= MAX_TOKENS:
            thinking_budget = MAX_TOKENS - 3
            logger.warning(
                f"THINKING_BUDGET is greater than MAX_TOKENS, setting it to MAX_TOKENS - 3: {thinking_budget}"
            )
        extra_body = {
            "reasoning_budget": thinking_budget,  # for nvidia api compatibility
            "thinking_token_budget": thinking_budget,  # for vllm compatibility
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }

    llm = NvidiaLLMService(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_LLM_URL,
        model=NVIDIA_LLM_MODEL,
        params=BaseOpenAILLMService.InputParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            **({"extra": {"extra_body": extra_body}} if enable_thinking else {}),
        ),
    )

    # ASR service config - add extended stop_history for multilingual mode
    stt_config = {
        "server": ASR_SERVER_URL,
        "api_key": NVIDIA_API_KEY,
        "language": ASR_LANGUAGE,
        "sample_rate": 16000,
        "generate_interruptions": VAD_PROFILE == VADProfile.ASR,
        "model": ASR_MODEL_NAME,
        "function_id": ASR_CLOUD_FUNCTION_ID,
    }
    if ENABLE_MULTILINGUAL:
        stt_config.update(stop_history=900, stop_history_eou=900)

    stt = NemotronASRService(**stt_config)

    # Load IPA dictionary with error handling
    ipa_file = IPA_FILE
    try:
        with open(ipa_file, encoding="utf-8") as f:
            ipa_dict = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"IPA dictionary file not found at {ipa_file}")
        raise FileNotFoundError(f"IPA dictionary file not found at {ipa_file}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in IPA dictionary file: {e}")
        raise ValueError(f"Invalid JSON in IPA dictionary file: {e}") from e
    except Exception as e:
        logger.error(f"Error loading IPA dictionary: {e}")
        raise

    # TTS text filter only enabled when ENABLE_TTS_TEXT_FILTER=true AND language is en-US
    enable_riva_text_filter = (
        ENABLE_TTS_TEXT_FILTER
        and TTS_LANGUAGE == "en-US"
        and not ENABLE_MULTILINGUAL
        and (SYSTEM_PROMPT_SELECTOR or "").lower() != "llama/tts_emotion_tags"
    )

    tts = NemotronTTSService(
        server=TTS_SERVER_URL,
        api_key=NVIDIA_API_KEY,
        voice_id=TTS_VOICE_ID,
        model=TTS_MODEL_NAME,
        language=TTS_LANGUAGE,
        sample_rate=22050,
        zero_shot_audio_prompt_file=(Path(ZERO_SHOT_AUDIO_PROMPT) if ZERO_SHOT_AUDIO_PROMPT else None),
        custom_dictionary=ipa_dict,
        text_filters=[RivaTextFilter()] if enable_riva_text_filter else [],
    )

    def _validated_selector(raw_value: str | None, default: str) -> str:
        selector = (raw_value or "").strip() or default
        if "/" not in selector:
            raise ValueError("SYSTEM_PROMPT_SELECTOR must be in '<model>/<prompt>' format")
        return selector

    if ENABLE_MULTILINGUAL:
        prompt_selector = _validated_selector(
            SYSTEM_PROMPT_SELECTOR,
            "llama-3.3-nemotron-super-49b-v1.5/multilingual_voice_assistant",
        )
        lang_codes = ", ".join(tts.list_available_voices().keys())
        messages = _resolve_prompt(prompt_selector)
        messages = [
            {"role": msg["role"], "content": _inject_prompt_variables(msg["content"], lang_codes=lang_codes)}
            for msg in messages
        ]
        logger.info(f"Loaded multilingual prompt: {prompt_selector} with languages: {lang_codes}")
    else:
        prompt_selector = _validated_selector(
            SYSTEM_PROMPT_SELECTOR,
            "nemotron-3-nano/generic_voice_assistant",
        )
        messages = _resolve_prompt(prompt_selector)
        logger.info(f"Loaded prompt: {prompt_selector}")

    # Defensive check to ensure the resolved prompt is not empty
    if not messages:
        raise ValueError(f"Resolved system prompt has no messages for selector: {prompt_selector}")

    if TALK_FIRST:
        messages.append({"role": "user", "content": "Hello"})

    # context = LLMContext(messages)
    context = OpenAILLMContext(messages=messages)

    if ENABLE_TOOL_CALLING:
        register_schema_tools_to_llm(llm, context, [GetCityWeatherTool()])

    logger.info(f"Context: {context}")
    logger.info(f"Messages: {messages}")
    logger.info(f"Tools: {context.tools}")

    enable_speculative_speech = ENABLE_SPECULATIVE_SPEECH
    chat_history_limit = CHAT_HISTORY_LIMIT

    # Preserve all initial prompt messages from prompt.yaml
    # This ensures system and first user messages (used for prompting) are never truncated
    preserve_prompt_messages = len(messages)

    if enable_speculative_speech:
        context_aggregator = create_nvidia_context_aggregator(
            context,
            send_interims=True,
            chat_history_limit=chat_history_limit,
            preserve_prompt_messages=preserve_prompt_messages,
        )
        tts_response_cacher = NvidiaTTSResponseCacher()
    else:
        context_aggregator = create_nvidia_context_aggregator(
            context,
            send_interims=False,
            chat_history_limit=chat_history_limit,
            preserve_prompt_messages=preserve_prompt_messages,
        )
        tts_response_cacher = None

    # Re-setup logging so the service initialization does not clobber loguru config.
    setup_rotating_log(log_file=LOG_FILE, log_level=LOG_LEVEL)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    original_messages = copy.deepcopy(messages)
    user_agg = context_aggregator.user()
    assistant_agg = context_aggregator.assistant()
    pipeline = Pipeline(
        [
            ws_transport.input(),  # WebSocket input from client
            rtvi,
            stt,  # Speech-To-Text
            user_agg,
            llm,  # LLM
            tts,  # Text-To-Speech
            *([tts_response_cacher] if tts_response_cacher else []),
            ws_transport.output(),  # WebSocket output to client
            assistant_agg,
        ]
    )

    resettable = []
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
            system_role="system",
            system_prompt_suffix="",
            enable_tool_calling=ENABLE_TOOL_CALLING,
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
            enable_metrics=True,
            enable_usage_metrics=True,
            idle_timeout=None,
        ),
        observers=[
            RTVIObserver(rtvi),
            UserBotLatencyLogObserver(),
        ],
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
        enable_tracing=IS_TRACING_ENABLED,
    )

    setup_rotating_log(log_file=LOG_FILE, log_level=LOG_LEVEL)

    await run_bot_websocket_server(
        task=task,
        ws_transport=ws_transport,
        rtvi=rtvi,
        task_ref=task_ref,
        audio_logger=None,
        talk_first=talk_first,
        initial_frame_factory=LLMRunFrame,
        on_disconnect_reset_services=resettable,
    )


app = create_fastapi_app(WEBSOCKET_PORT)


async def main():
    logger.info(
        f"Starting servers with host {SERVER_HOST}, "
        f"WebSocket on port {WEBSOCKET_PORT}, FastAPI on port {FASTAPI_PORT}"
    )
    await run_bot_with_fastapi(
        ws_coro=run_bot_websocket(
            host=SERVER_HOST,
            port=WEBSOCKET_PORT,
        ),
        app=app,
        host=SERVER_HOST,
        fastapi_port=FASTAPI_PORT,
    )


if __name__ == "__main__":
    asyncio.run(main())
