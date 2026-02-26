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

"""
Voice Agent Evaluation Bridge

Connects two voice agents via WebSocket and provides:
- Bidirectional audio routing
- Response latency measurement
- Dynamic system prompt updates via RTVI actions
- Conversation monitoring and metrics
"""
import asyncio
import json
import queue
import random
import threading
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import soxr
import websockets
from loguru import logger
from omegaconf import DictConfig
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.processors.frameworks.rtvi import (
    RTVIBotStartedSpeakingMessage,
    RTVIBotStoppedSpeakingMessage,
    RTVIBotTranscriptionMessage,
    RTVIBotTTSTextMessage,
    RTVIServerMessage,
    RTVITextMessageData,
)
from pipecat.serializers.protobuf import MessageFrame, ProtobufFrameSerializer

from nemo.agents.voice_agent.evaluation.tools.rtvi_control import (
    EXIT_MESSAGE_END_TAG,
    EXIT_MESSAGE_START_TAG,
    FINAL_RESPONSE_END_TAG,
    FINAL_RESPONSE_START_TAG,
)
from nemo.agents.voice_agent.utils import setup_logging

# Import AudioStream for buffering and resampling
from nemo.agents.voice_agent.utils.audio import AudioStream, NoiseConfig

# RTVI message type constants - automatically adapts to pipecat changes
RTVI_BOT_STOPPED_SPEAKING = RTVIBotStoppedSpeakingMessage().type
RTVI_BOT_STARTED_SPEAKING = RTVIBotStartedSpeakingMessage().type
RTVI_BOT_TRANSCRIPTION = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text="")).type
RTVI_BOT_TTS_TEXT = RTVIBotTTSTextMessage(data=RTVITextMessageData(text="")).type
RTVI_BOT_SERVER_MESSAGE = RTVIServerMessage(data=RTVITextMessageData(text="")).type

STOP_REASON_TIMEOUT = "[TIMEOUT]"
STOP_REASON_EXIT = "[EXIT]"


@dataclass
class ResponseLatency:
    """Single response latency measurement"""

    user_stop_time: float  # When user stopped speaking
    agent_start_time: float  # When agent started responding
    latency_ms: float  # Response latency in milliseconds
    user_transcript: str = ""
    agent_transcript: str = ""


@dataclass
class SegmentEntry:
    """Entry for segLST format (segment list with timing)"""

    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    speaker: str  # "user" or "agent"
    transcript: str  # Text content


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation"""

    turns: list = field(default_factory=list)
    latencies: List[ResponseLatency] = field(default_factory=list)
    start_time: datetime = None
    end_time: datetime = None

    # Buffered log entries (start_time, formatted_entry) - sorted and written at end
    log_entries: List[Tuple[float, str]] = field(default_factory=list)

    # Audio timing state
    user_last_audio_time: Optional[float] = None
    agent_last_audio_time: Optional[float] = None
    waiting_for_agent_response: bool = False
    last_user_transcript: str = ""

    # Transcript accumulation (segments arrive incrementally)
    user_current_transcript: str = ""
    agent_current_transcript: str = ""

    thread_start_timestamp: Optional[float] = None  # When routing threads start (for conversation log timing)

    # Segment tracking for segLST output
    segments: List[SegmentEntry] = field(default_factory=list)
    current_user_segment: Optional[SegmentEntry] = None
    current_agent_segment: Optional[SegmentEntry] = None

    agent_final_response: List[str] = field(default_factory=list)
    agent_final_response_time: List[float] = field(default_factory=list)

    def get_latency_stats(self):
        """Calculate latency statistics"""
        if not self.latencies:
            return {
                "count": 0,
                "mean_ms": 0,
                "p50_ms": 0,
                "p95_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
            }

        latencies_sorted = sorted([l.latency_ms for l in self.latencies])
        count = len(latencies_sorted)

        return {
            "count": count,
            "mean_ms": sum(latencies_sorted) / count,
            "p50_ms": latencies_sorted[count // 2],
            "p95_ms": latencies_sorted[int(count * 0.95)] if count > 0 else 0,
            "min_ms": latencies_sorted[0],
            "max_ms": latencies_sorted[-1],
        }

    def reset(self):
        """Reset all metrics to prepare for a new scenario"""
        self.start_time = None
        self.end_time = None

        # Reset latency tracking state
        self.user_last_audio_time = None
        self.agent_last_audio_time = None
        self.waiting_for_agent_response = False
        self.last_user_transcript = ""

        # Clear accumulated transcript segments
        self.user_current_transcript = ""
        self.agent_current_transcript = ""

        # Reset scenario-specific metrics (for multi-scenario evaluations)
        self.latencies = []
        self.turns = []
        self.segments = []
        self.log_entries = []

        self.thread_start_timestamp = None
        self.current_user_segment = None
        self.current_agent_segment = None

        self.agent_final_response = []
        self.agent_final_response_time = []


class VoiceAgentEvaluationBridge:
    """
    Evaluation bridge that connects two voice agents via WebSocket
    and provides control through RTVI actions.

    Key features:
    - Routes audio bidirectionally between agents
    - Monitors transcriptions and metrics
    - Measures response latency by tracking audio frames
    - Can send RTVI control messages to update prompts
    - Works with distributed agents
    """

    def __init__(
        self,
        user_url: str,
        agent_url: str,
        output_dir: Optional[str] = None,
        scenario_name: Optional[str] = None,
        user_output_sample_rate: int = 24000,
        agent_output_sample_rate: int = 24000,
        user_input_sample_rate: int = 16000,
        agent_input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
        audio_chunk_in_seconds: float = 0.016,
        use_burst_mode: bool = False,
        burst_size_range: Tuple[int, int] = (3, 8),
        burst_delay_ms: int = 0,
        grace_period: float = 1.0,
        turn_start_offset_secs: float = -0.0,
        turn_end_offset_secs: float = -0.3,
        noise_config: Optional[NoiseConfig] = None,
        log_level: str = "DEBUG",
    ):
        """
        Args:
            user_url: URL of the user WebSocket
            agent_url: URL of the agent WebSocket
            output_dir: Directory for all output files (conversation log, audio, segLST)
            scenario_name: Name of the scenario
            user_output_sample_rate: Sample rate of the user output
            agent_output_sample_rate: Sample rate of the agent output
            user_input_sample_rate: Sample rate of the user input
            agent_input_sample_rate: Sample rate of the agent input
            output_sample_rate: Sample rate of the output
            audio_chunk_in_seconds: Duration of the audio chunk in seconds
            use_burst_mode: Whether to use burst mode, default to steady mode with fixed interval
            burst_size_range: Range of the random burst size, used to simulate the irregular sending pattern of a browser.
            burst_delay_ms: Delay between the frames in the random burst, used to simulate the irregular sending pattern of a browser.
            grace_period: Grace period after the main duration, used to drain the websocket
            turn_start_offset_secs: Offset added to turn start times in conversation log and segLST,
                so that the latency by BOT_STARTED_SPEAKING event is mitigated. This is a workaround to the fact
                that the BOT_STARTED_SPEAKING event may come after the first audio chunk is sent.
            turn_end_offset_secs: Offset added to turn end times in conversation log and segLST,
                so that the latency by BOT_STOPPED_SPEAKING event is mitigated. This is a workaround to the fact
                that the BOT_STOPPED_SPEAKING event is sent after 0.35s silence in Pipecat output transport.
            noise_config: Noise configuration, used to configure the noise for the audio stream
        """
        self.user_url = user_url
        self.agent_url = agent_url
        self.output_dir = output_dir
        self.scenario_name = scenario_name
        self.log_file = None
        self.seglst_file = None
        self.bridge_audio_file = None
        self.user_output_sample_rate = user_output_sample_rate
        self.agent_output_sample_rate = agent_output_sample_rate
        self.user_input_sample_rate = user_input_sample_rate
        self.agent_input_sample_rate = agent_input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.audio_chunk_in_seconds = audio_chunk_in_seconds
        self.log_level = log_level

        # Random burst mode configuration (simulates browser's irregular sending pattern)
        self.use_burst_mode = use_burst_mode  # Disable burst mode by default
        self.burst_size_range = burst_size_range  # Random frames per burst
        self.burst_delay_ms = burst_delay_ms  # sleep duration between frames in burst
        # Pause calculated per burst: (burst_size × 16ms) - burst_duration
        # This maintains 16ms average per frame while varying the pattern

        # Grace period and timeout configuration for send loops
        self.grace_period = grace_period  # Extra time to drain audio after main duration

        self.turn_start_offset_secs = turn_start_offset_secs
        self.turn_end_offset_secs = turn_end_offset_secs

        # Noise configuration for user channel
        self.noise_config = noise_config

        self.user_ws = None
        self.agent_ws = None

        self.metrics = EvaluationMetrics()

        # Serializers for protobuf communication
        self.serializer = ProtobufFrameSerializer()

        # Track RTVI state
        self.user_ready = False
        self.agent_ready = False

        # Debug: accumulate sent audio chunks for analysis (only final sent audio)
        self.sent_to_agent_chunks = []  # USER→AGENT final sent chunks
        self.sent_to_user_chunks = []  # AGENT→USER final sent chunks

        # Thread-safe queues for audio routing between threads
        # Each queue passes raw audio bytes between WebSocket threads
        self.user_to_agent_queue = queue.Queue()  # User audio → Agent
        self.agent_to_user_queue = queue.Queue()  # Agent audio → User

        # Thread control
        self.stop_event = threading.Event()
        self.threads = []
        self.stop_reason = STOP_REASON_TIMEOUT

        # Bridge resamples at source (like browser client) for better quality
        # This avoids STT having to resample small chunks
        logger.info("Bridge configured to resample audio at source (simulating browser behavior)")
        logger.info(f"  User: {self.user_output_sample_rate}Hz (TTS) → {self.agent_input_sample_rate}Hz (STT)")
        logger.info(f"  Agent: {self.agent_output_sample_rate}Hz (TTS) → {self.user_input_sample_rate}Hz (STT)")

        # Log burst mode configuration
        if self.use_burst_mode:
            logger.info(
                f"Random burst mode enabled: {self.burst_size_range[0]}-{self.burst_size_range[1]} frames per burst, {self.burst_delay_ms}ms between frames"
            )
            min_pause = (self.burst_size_range[0] * self.audio_chunk_in_seconds * 1000) - (
                (self.burst_size_range[0] - 1) * self.burst_delay_ms
            )
            max_pause = (self.burst_size_range[1] * self.audio_chunk_in_seconds * 1000) - (
                (self.burst_size_range[1] - 1) * self.burst_delay_ms
            )
            logger.info(f"  Pause range: {min_pause:.0f}-{max_pause:.0f}ms (calculated to maintain 16ms avg)")
        else:
            logger.info(f"Steady mode: sending at constant {self.audio_chunk_in_seconds * 1000:.0f}ms intervals")

        # Initialize output directory and log files
        if output_dir:
            self.init_output_dir(output_dir, scenario_name, log_level)

        self.bridge_ready = False
        self.needs_reset = False
        self.final_response_file = "final_agent_response.json"
        self.user_context_history = None
        self.agent_context_history = None

    def init_output_dir(self, output_dir: str, scenario_name: Optional[str] = None, log_level: str = "DEBUG"):
        """Initialize the output directory and all derived log/audio file paths."""
        logger.info(f"Initializing output directory: {output_dir}, session name: {scenario_name}")
        self.output_dir = output_dir
        self.scenario_name = scenario_name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = str(Path(output_dir) / "conversation_log.txt")
        self.seglst_file = str(Path(output_dir) / "conversation_log.seglst.json")
        self.bridge_audio_file = str(Path(output_dir) / "conversation_log.wav")

        # Initialize logging for this scenario
        bridge_log_file = str(Path(output_dir) / "bridge_log.txt")
        setup_logging(log_file=bridge_log_file, log_level=log_level)  # Update logging to write to this file

        try:
            with open(self.log_file, "w") as f:
                f.write("RTVI Evaluation Bridge - Conversation Log\n")
                f.write("=" * 80 + "\n")
                f.write(f"Start Time: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            logger.error(f"Error initializing log file: {e}")
            return False
        return True

    def set_noise_config(self, noise_config: Optional[Union[NoiseConfig, dict]] = None):
        """Set the noise configuration"""
        logger.info(f"Setting noise configuration: {noise_config}")
        if noise_config is not None:
            if isinstance(noise_config, dict):
                noise_config = NoiseConfig(**noise_config)
            self.noise_config = noise_config
        else:
            self.noise_config = None

    async def prepare_for_scenario(self, scenario: Union[dict, DictConfig], output_dir: str, log_level: str = "DEBUG"):
        """Prepare the bridge for a scenario"""

        # Initialize output directory for this scenario
        self.init_output_dir(output_dir, scenario_name=scenario['name'], log_level=log_level)

        # Reset bridge before each scenario, and create connection to update the prompts
        await self.connect()

        # Update prompts (handler will automatically reset)
        await self.update_user_prompt(prompt=scenario["user_prompt"], tools=scenario["user_tools"])
        await self.update_agent_prompt(prompt=scenario["agent_prompt"], tools=scenario["agent_tools"])

        if "noise_config" in scenario:
            self.set_noise_config(scenario["noise_config"])
        else:
            self.set_noise_config(None)

        # Disconnect the bridge to clear the WebSocket buffers
        await self.disconnect(print_stats=False)

        logger.info(f"Finished preparing for scenario: {scenario['name']}")
        self.bridge_ready = True

    def _get_relative_time(self, timestamp: float) -> float:
        """
        Get time relative to scenario start (thread start time).

        Args:
            timestamp: Absolute timestamp (asyncio loop time)

        Returns:
            Time in seconds relative to thread_start_timestamp, or 0 if not set
        """
        if self.metrics.thread_start_timestamp is None:
            return 0.0
        return timestamp - self.metrics.thread_start_timestamp

    def _finalize_speaker_turn(self, speaker: str, timestamp: float) -> Optional[SegmentEntry]:
        """
        Finalize the current in-progress turn for the given speaker.

        Sets end_time, assigns transcript (or "[INTERRUPTED]" if no TTS text was received),
        appends the segment to self.metrics.segments, and clears accumulation state.

        Args:
            speaker: "user" or "agent"
            timestamp: Absolute timestamp (asyncio loop time)

        Returns:
            The finalized SegmentEntry, or None if no segment was in progress.
        """
        if speaker == "user":
            segment = self.metrics.current_user_segment
            transcript_acc = self.metrics.user_current_transcript
        else:
            segment = self.metrics.current_agent_segment
            transcript_acc = self.metrics.agent_current_transcript

        if segment is None:
            return None

        transcript = transcript_acc.strip() or "[INTERRUPTED]"
        segment.end_time = self._get_relative_time(timestamp)
        segment.transcript = transcript
        self.metrics.segments.append(segment)

        # Clear state
        if speaker == "user":
            self.metrics.current_user_segment = None
            self.metrics.user_current_transcript = ""
        else:
            self.metrics.current_agent_segment = None
            self.metrics.agent_current_transcript = ""

        logger.info(f"[{speaker.upper()}] {transcript}")
        return segment

    def _format_turn_log(
        self, role: str, text: str, start_time: float, end_time: float, latency_ms: float = None
    ) -> str:
        """
        Format a turn entry for the conversation log.

        Args:
            role: "user" or "agent"
            text: Transcript text
            start_time: Turn start time (relative to scenario start)
            end_time: Turn end time (relative to scenario start)
            latency_ms: Optional response latency in milliseconds

        Returns:
            Formatted log entry string
        """
        duration = end_time - start_time
        log_entry = f"[{start_time:7.3f}s - {end_time:7.3f}s] ({duration:.3f}s) {role.upper()}: {text}\n"
        if latency_ms is not None:
            log_entry += f"  → Response latency: {latency_ms:.1f}ms\n"
        return log_entry

    async def connect(self, max_retries: int = 5, retry_delay: float = 1.0):
        """Connect to both user and agent with retry logic

        Args:
            max_retries: Maximum number of connection attempts per endpoint
            retry_delay: Initial delay between retries (doubles each retry)
        """
        # Connect to user with retries
        logger.info(f"Connecting to user at {self.user_url}")
        for attempt in range(max_retries):
            try:
                self.user_ws = await websockets.connect(
                    self.user_url, ping_interval=20, ping_timeout=10, close_timeout=10
                )
                logger.info(f"User connection established (attempt {attempt + 1})")
                break
            except (OSError, websockets.exceptions.WebSocketException) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(f"User connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"User connection failed after {max_retries} attempts")
                    raise

        # Connect to agent with retries
        logger.info(f"Connecting to agent at {self.agent_url}")
        for attempt in range(max_retries):
            try:
                self.agent_ws = await websockets.connect(
                    self.agent_url, ping_interval=20, ping_timeout=10, close_timeout=10
                )
                logger.info(f"Agent connection established (attempt {attempt + 1})")
                break
            except (OSError, websockets.exceptions.WebSocketException) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(f"Agent connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Agent connection failed after {max_retries} attempts")
                    raise

        # Send RTVI client-ready handshake to both agents
        await self._send_client_ready(self.user_ws)
        await self._send_client_ready(self.agent_ws)
        await self.reset()
        logger.info("Both agents connected and ready")

    async def _send_client_ready(self, ws):
        """Send RTVI client-ready handshake and wait for bot-ready"""
        client_ready_msg = {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": f"client_ready_{datetime.now().timestamp()}",
            "data": {"version": "1.1.0", "about": {"library": "evaluation-bridge", "library_version": "1.0.0"}},
        }

        # Serialize as MessageFrame and send
        msg_frame = MessageFrame(data=json.dumps(client_ready_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await ws.send(serialized)

        logger.info("Client-ready handshake sent, waiting for bot-ready...")

        # Wait for bot-ready response
        try:
            timeout = 5.0
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    if isinstance(msg, bytes):
                        frame = await self.serializer.deserialize(msg)
                        if hasattr(frame, 'message') and frame.message:
                            if isinstance(frame.message, str):
                                data = json.loads(frame.message)
                            else:
                                data = frame.message

                            if data.get("type") == "bot-ready":
                                logger.info("Received bot-ready response")
                                return True
                except asyncio.TimeoutError:
                    continue

            logger.warning("Timeout waiting for bot-ready response")
            return False
        except Exception as e:
            logger.error(f"Error waiting for bot-ready: {e}")
            return False

    async def update_user_prompt(self, prompt: str, tools: str, auto_reset: bool = False, add_suffix: bool = False):
        """
        Update user's system prompt via RTVI action.

        Args:
            prompt: New system prompt text
            tools: New tools in json string format
            auto_reset: If True, also sends reset action after updating prompt
            add_suffix: If True, add previously configured system prompt suffix to the new prompt
        """
        logger.info(f"Updating user prompt: {prompt[:100]}..., tools: {tools[:100]}...")

        # Create RTVI action message
        action_msg = {
            "label": "rtvi-ai",
            "type": "action",
            "id": f"update_prompt_{datetime.now().timestamp()}",
            "data": {
                "service": "context",
                "action": "update_system_prompt",
                "arguments": [
                    {"name": "prompt", "value": prompt},
                    {"name": "tools", "value": tools},
                    {"name": "add_suffix", "value": add_suffix},
                ],
            },
        }

        # Serialize as MessageFrame and send
        msg_frame = MessageFrame(data=json.dumps(action_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await self.user_ws.send(serialized)

        logger.info("User prompt update sent")

        if auto_reset:
            logger.info("Sending additional reset action to user...")
            await self._send_reset_action(self.user_ws, "user")

        return True

    async def update_agent_prompt(self, prompt: str, tools: str, auto_reset: bool = False, add_suffix: bool = False):
        """
        Update agent's system prompt via RTVI action.

        Args:
            prompt: New system prompt text
            tools: New tools in json string format
            auto_reset: If True, also sends reset action after updating prompt
            add_suffix: If True, add previously configured system prompt suffix to the new prompt
        """
        logger.info(f"Updating agent prompt: {prompt[:100]}..., tools: {tools[:100]}...")

        # Create RTVI action message
        action_msg = {
            "label": "rtvi-ai",
            "type": "action",
            "id": f"update_prompt_{datetime.now().timestamp()}",
            "data": {
                "service": "context",
                "action": "update_system_prompt",
                "arguments": [
                    {"name": "prompt", "value": prompt},
                    {"name": "tools", "value": tools},
                    {"name": "add_suffix", "value": add_suffix},
                ],
            },
        }

        # Serialize as MessageFrame and send
        msg_frame = MessageFrame(data=json.dumps(action_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await self.agent_ws.send(serialized)

        logger.info("Agent prompt update sent")

        if auto_reset:
            logger.info("Sending additional reset action to agent...")
            await self._send_reset_action(self.agent_ws, "agent")

        return True

    async def _send_reset_action(self, ws, agent_name: str):
        """
        Send RTVI reset action to clear conversation history.

        Args:
            ws: WebSocket connection
            agent_name: Name of agent (for logging)
        """
        if not ws:
            logger.info(f"[{agent_name.capitalize()}] Websocket is not connected, skipping reset")
            return

        reset_msg = {
            "label": "rtvi-ai",
            "type": "action",
            "id": f"reset_{datetime.now().timestamp()}",
            "data": {
                "service": "context",
                "action": "reset",
                "arguments": [],
            },
        }

        # Serialize as MessageFrame and send
        msg_frame = MessageFrame(data=json.dumps(reset_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await ws.send(serialized)

        logger.info(f"{agent_name.capitalize()} reset action sent")

    async def reset(self):
        """
        Reset metrics and both agents' conversation history
        """
        logger.info("Resetting metrics and conversation context...")
        await self.reset_user()
        await self.reset_agent()
        # Reset all metrics
        self.metrics.reset()
        self.needs_reset = False

    async def reset_agent(self):
        """
        Reset agent's conversation history.
        Useful to clear context between evaluation scenarios.
        """
        if self.agent_ws:
            logger.info("Resetting agent...")
            await self._send_reset_action(self.agent_ws, "agent")
            logger.info("Agent reset complete")

    async def reset_user(self):
        """
        Reset user's conversation history.
        Useful to clear context between evaluation scenarios.
        """
        if self.user_ws:
            logger.info("Resetting user...")
            await self._send_reset_action(self.user_ws, "user")
            logger.info("User reset complete")

    async def send_text_to_user(self, text: str):
        """
        Send a text message to the user agent to trigger conversation.

        Args:
            text: Text to send to user agent's LLM
        """
        send_text_msg = {
            "label": "rtvi-ai",
            "type": "send-text",
            "id": f"send_text_{datetime.now().timestamp()}",
            "data": {"content": text, "options": {"run_immediately": True, "audio_response": True}},
        }

        msg_frame = MessageFrame(data=json.dumps(send_text_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await self.user_ws.send(serialized)

        logger.info(f"Sent text to user: {text[:50]}...")

    async def send_text_to_agent(self, text: str):
        """
        Send a text message to the agent agent to trigger conversation.

        Args:
            text: Text to send to agent agent's LLM
        """
        send_text_msg = {
            "label": "rtvi-ai",
            "type": "send-text",
            "id": f"send_text_{datetime.now().timestamp()}",
            "data": {"content": text, "options": {"run_immediately": True, "audio_response": True}},
        }

        msg_frame = MessageFrame(data=json.dumps(send_text_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await self.agent_ws.send(serialized)

        logger.info(f"Sent text to agent: {text[:50]}...")

    async def _wait_for_action_response(self, ws, timeout=5.0):
        """Wait for RTVI action response"""
        try:
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)

                    if isinstance(msg, str):
                        data = json.loads(msg)

                        if data.get("data", {}).get("message_type") == "action-response":
                            result = data.get("data", {}).get("result", {})
                            return result.get("success", False) or result is True
                except asyncio.TimeoutError:
                    continue

            logger.warning("Timeout waiting for action response")
            return False
        except Exception as e:
            logger.error(f"Error waiting for response: {e}")
            return False

    async def _receive_to_queue(
        self,
        ws: websockets.WebSocketClientProtocol,
        duration: float,
        direction: str,
        queue: queue.Queue,
        monitor_func: Callable,
    ):
        """
        Receive audio from websocket and put into queue.

        Args:
            ws: Source websocket to receive from
            duration: How long to run the receive loop in seconds
            direction: For logging (e.g., "USER→AGENT", "AGENT→USER")
            queue: Thread-safe queue to put audio chunks into
            monitor_func: Async monitoring function for metrics (e.g., _monitor_user_message)
        """
        logger.info(f"[{direction}] Starting receive loop")
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        try:
            while not self.stop_event.is_set():
                # Use short timeout so we can check stop_event periodically
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                # Deserialize frame
                try:
                    frame = await self.serializer.deserialize(message)
                    if frame is None:
                        continue
                except Exception as e:
                    logger.error(f"[{direction}] Deserialization error: {e}")
                    continue

                current_time = loop.time()
                elapsed = current_time - start_time

                # Check if we're past the main duration
                in_grace_period = elapsed > duration
                if in_grace_period:
                    # logger.debug(f"[{direction}] In grace period, skip monitoring message: {frame}")
                    continue

                # Monitor messages
                await monitor_func(frame)

                # Check if this is audio
                if hasattr(frame, 'audio') and frame.audio:
                    # Put raw audio into thread-safe queue
                    queue.put(frame.audio)
                    logger.debug(f"[{direction}] Queued {len(frame.audio)} bytes of audio")

            if self.stop_event.is_set():
                logger.info(f"[{direction}] Stop event received, exiting receive loop")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[{direction}] Receive WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"[{direction}] Receive error: {e}", exc_info=True)

    async def _send_audio_stream(
        self,
        audio_stream: AudioStream,
        dest_ws: websockets.WebSocketClientProtocol,
        direction: str,
        duration: int,
        source_queue: queue.Queue,
        sent_chunks_list: List[bytes],
    ):
        """
        Send audio stream at fixed intervals from AudioStream with duration and grace period.

        Args:
            audio_stream: AudioStream containing buffered and resampled audio
            dest_ws: Destination websocket to send to
            direction: For logging (e.g., "USER→AGENT", "AGENT→USER")
            duration: How long to run the send loop in seconds
            source_queue: Queue to retrieve audio chunks from
            sent_chunks_list: List to append sent chunks to for tracking
        """
        logger.info(f"[{direction}] Starting send loop")

        loop = asyncio.get_event_loop()
        start_time = loop.time()
        target_time = start_time  # Track target time incrementally for numerical stability

        try:
            while not self.stop_event.is_set():
                current_time = loop.time()
                elapsed = current_time - start_time
                in_grace_period = elapsed > duration
                if elapsed > (duration + self.grace_period):
                    logger.info(f"[{direction}] Grace period expired after {elapsed:.1f}s, stopping")
                    break

                # Empty all available audio from thread-safe queue (non-blocking)
                # This prevents stale audio buildup during burst pauses or LLM/TTS blocking
                chunks_retrieved = 0
                while True:
                    try:
                        audio_chunk = source_queue.get_nowait()
                        # Put into AudioStream for buffering/resampling
                        await audio_stream.put(audio_chunk)
                        chunks_retrieved += 1
                    except queue.Empty:
                        break

                if chunks_retrieved > 0:
                    logger.debug(f"[{direction}] Retrieved {chunks_retrieved} chunks from queue")
                if in_grace_period:
                    # logger.debug(f"[{direction}] In grace period, skip forwarding audio: {chunks_retrieved} chunks")
                    continue

                # Burst sending: send N frames rapidly, then pause
                # Steady mode is just burst_size=1 (send 1 frame, pause 16ms, repeat)
                burst_size = (
                    random.randint(self.burst_size_range[0], self.burst_size_range[1]) if self.use_burst_mode else 1
                )

                # Send burst frames
                for idx in range(burst_size):
                    if idx > 0 and self.burst_delay_ms > 0:
                        # Small delay between frames in burst
                        await asyncio.sleep(self.burst_delay_ms / 1000.0)

                    # Get audio from AudioStream
                    audio_to_send, has_speech = await audio_stream.get_nowait()

                    # Track sent audio
                    sent_chunks_list.append(audio_to_send)

                    # Create frame and send
                    output_frame = OutputAudioRawFrame(
                        audio=audio_to_send, sample_rate=audio_stream.output_sample_rate, num_channels=1
                    )
                    serialized = await self.serializer.serialize(output_frame)
                    await dest_ws.send(serialized)

                    # if has_speech:
                    #     logger.debug(
                    #         f"[{direction}] Sent {len(audio_to_send)} bytes ({idx+1}/{burst_size}, has_speech: {has_speech})"
                    #     )

                # Time-based scheduling: increment target time from previous burst
                # This automatically compensates for processing overhead and is numerically stable
                target_time += burst_size * self.audio_chunk_in_seconds
                current_time = loop.time()
                wait_duration = max(0.001, target_time - current_time)

                if wait_duration < 0.001:
                    logger.debug(f"[{direction}] Behind schedule by {-wait_duration:.3f}s")

                if self.use_burst_mode:
                    logger.debug(
                        f"[{direction}] Burst complete ({burst_size} frames), waiting {wait_duration*1000:.1f}ms (target: {target_time:.3f}s)"
                    )
                await asyncio.sleep(wait_duration)

            if self.stop_event.is_set():
                logger.info(f"[{direction}] Stop event received, exiting send loop")
            else:
                logger.info(f"[{direction}] Send loop finished")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[{direction}] WebSocket closed: {e}")
        except Exception as e:
            # print traceback
            import traceback

            traceback.print_exc()
            logger.error(f"[{direction}] Send error: {e}", exc_info=True)

    def user_websocket_thread(self, duration: int):
        """
        Thread 1: Handle all user WebSocket traffic (bidirectional).

        This thread:
        - Receives audio from user WebSocket
        - Puts user audio into user_to_agent_queue for agent thread
        - Gets agent audio from agent_to_user_queue
        - Sends agent audio to user WebSocket

        Args:
            duration: How long to run (seconds)
        """
        logger.info("[USER THREAD] Starting user WebSocket thread")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def user_loop():
            try:
                # Connect to user WebSocket
                async with websockets.connect(self.user_url) as user_ws:
                    self.user_ws = user_ws
                    logger.info(f"[USER THREAD] Connected to user: {self.user_url}")

                    # Wait for ready handshake
                    await self._send_client_ready(user_ws)

                    # Create AudioStream for agent→user (buffering and resampling)
                    agent_to_user_stream = AudioStream(
                        chunk_size_in_seconds=self.audio_chunk_in_seconds,
                        input_sample_rate=self.agent_output_sample_rate,
                        output_sample_rate=self.user_input_sample_rate,
                        stream_resampler=False,
                        tag="AGENT→USER",
                    )

                    # Run bidirectional tasks (send loop manages timeout + grace period)
                    # Add overall timeout with grace period to stop receive loop when send loop finishes
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(
                                # Receive from user, put raw audio into queue
                                self._receive_user_to_queue(user_ws, duration),
                                # Get raw audio from queue, send to user (handles its own timeout)
                                self._send_agent_to_user(user_ws, agent_to_user_stream, duration),
                            ),
                            timeout=duration + self.grace_period,  # Extra 1s buffer for cleanup
                        )
                    except asyncio.TimeoutError:
                        logger.info("[USER THREAD] Overall timeout reached, stopping receive loop")

                    # at the end, send an RTVI message to the user to tell it to return the context history
                    self.user_context_history = await self._retrieve_context_history(user_ws)

            except Exception as e:
                logger.error(f"[USER THREAD] Error: {e}", exc_info=True)
            finally:
                logger.info("[USER THREAD] Exiting")

        try:
            loop.run_until_complete(user_loop())
        finally:
            loop.close()

    def agent_websocket_thread(self, duration: int):
        """
        Thread 2: Handle all agent WebSocket traffic (bidirectional).

        This thread:
        - Gets user audio from user_to_agent_queue
        - Sends user audio to agent WebSocket
        - Receives audio from agent WebSocket
        - Puts agent audio into agent_to_user_queue for user thread

        Args:
            duration: How long to run (seconds)
        """
        logger.info("[AGENT THREAD] Starting agent WebSocket thread")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def agent_loop():
            try:
                # Connect to agent WebSocket
                async with websockets.connect(self.agent_url) as agent_ws:
                    self.agent_ws = agent_ws
                    logger.info(f"[AGENT THREAD] Connected to agent: {self.agent_url}")

                    # Wait for ready handshake
                    await self._send_client_ready(agent_ws)

                    # Create AudioStream for user→agent (buffering and resampling)
                    user_to_agent_stream = AudioStream(
                        chunk_size_in_seconds=self.audio_chunk_in_seconds,
                        input_sample_rate=self.user_output_sample_rate,
                        output_sample_rate=self.agent_input_sample_rate,
                        stream_resampler=False,
                        tag="USER→AGENT",
                        noise_config=self.noise_config,
                    )

                    # Send kickoff message after a delay
                    async def send_kickoff():
                        await asyncio.sleep(1)
                        logger.info("[AGENT THREAD] Sending kickoff message to agent...")
                        await self.send_text_to_agent("Hello")

                    # Run bidirectional tasks (send loop manages timeout + grace period)
                    # Add overall timeout with grace period to stop receive loop when send loop finishes
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(
                                # Get raw audio from queue, send to agent (handles its own timeout)
                                self._send_user_to_agent(agent_ws, user_to_agent_stream, duration),
                                # Receive from agent, put raw audio into queue
                                self._receive_agent_to_queue(agent_ws, duration),
                                send_kickoff(),
                            ),
                            timeout=duration + self.grace_period,
                        )
                    except asyncio.TimeoutError:
                        logger.info("[AGENT THREAD] Overall timeout reached, stopping receive loop")

                    # at the end, send an RTVI message to the agent to tell it to return the context history
                    self.agent_context_history = await self._retrieve_context_history(agent_ws)

            except Exception as e:
                logger.error(f"[AGENT THREAD] Error: {e}", exc_info=True)
            finally:
                logger.info("[AGENT THREAD] Exiting")

        try:
            loop.run_until_complete(agent_loop())
        finally:
            loop.close()

    async def _retrieve_context_history(self, ws) -> dict:
        """
        Retrieve the context history from the WebSocket. First send a message to the ws to trigger the
        `get_context_history` RTVI action, then wait for the response.
        Args:
            ws: WebSocket connection
        Returns:
            context_history: context history as a dictionary with two keys: `context` and `logs`,
                where `context` the LLM context history, and `logs` is the bot server logs.
        """
        if not ws:
            logger.warning("[CONTEXT HISTORY] WebSocket is not connected, skipping context history retrieval")
            return {}

        try:
            action_msg = {
                "label": "rtvi-ai",
                "type": "action",
                "id": f"get_context_history_{datetime.now().timestamp()}",
                "data": {
                    "service": "context",
                    "action": "get_context_history",
                    "arguments": [],
                },
            }

            # Serialize as MessageFrame and send
            msg_frame = MessageFrame(data=json.dumps(action_msg))
            serialized = await self.serializer.serialize(msg_frame)
            await ws.send(serialized)

            logger.info("[CONTEXT HISTORY] Sent get_context_history action, waiting for response...")

            # Wait for the action-response with a longer timeout since log content can be large
            timeout = 15.0
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    # Deserialize the protobuf frame
                    frame = await self.serializer.deserialize(msg)
                    if frame is None:
                        continue

                    # Extract message data from the frame
                    if not (hasattr(frame, 'message') and frame.message):
                        continue

                    data = json.loads(frame.message) if isinstance(frame.message, str) else frame.message

                    if data.get("type") == "action-response":
                        result = data.get("data", {}).get("result", {})
                        logger.info(
                            f"[CONTEXT HISTORY] Received context history "
                            f"(context: {len(result.get('context', []))} messages, "
                            f"logs: {len(result.get('logs', ''))} chars)"
                        )
                        return result
                except asyncio.TimeoutError:
                    continue

            logger.warning("[CONTEXT HISTORY] Timeout waiting for context history response")
            return {}
        except Exception as e:
            logger.warning(f"[CONTEXT HISTORY] Error retrieving context history: {e}")
            return {}

    async def _receive_user_to_queue(self, user_ws, duration: float):
        """Receive audio from user WebSocket and put into queue for agent thread."""
        return await self._receive_to_queue(
            ws=user_ws,
            duration=duration,
            direction="USER→AGENT",
            queue=self.user_to_agent_queue,
            monitor_func=self._monitor_user_message,
        )

    async def _send_agent_to_user(self, user_ws, audio_stream: AudioStream, duration: int):
        """Get audio from queue, process through AudioStream, send to user WebSocket."""
        return await self._send_audio_stream(
            audio_stream=audio_stream,
            dest_ws=user_ws,
            direction="AGENT→USER",
            duration=duration,
            source_queue=self.agent_to_user_queue,
            sent_chunks_list=self.sent_to_user_chunks,
        )

    async def _send_user_to_agent(self, agent_ws, audio_stream: AudioStream, duration: int):
        """Get audio from queue, process through AudioStream, send to agent WebSocket."""
        return await self._send_audio_stream(
            audio_stream=audio_stream,
            dest_ws=agent_ws,
            direction="USER→AGENT",
            duration=duration,
            source_queue=self.user_to_agent_queue,
            sent_chunks_list=self.sent_to_agent_chunks,
        )

    async def _receive_agent_to_queue(self, agent_ws, duration: float):
        """Receive audio from agent WebSocket and put into queue for user thread."""
        return await self._receive_to_queue(
            ws=agent_ws,
            duration=duration,
            direction="AGENT→USER",
            queue=self.agent_to_user_queue,
            monitor_func=self._monitor_agent_message,
        )

    async def run_scenario(self, duration: int = 300):
        """
        Route audio between agents and monitor conversation.
        Uses separate threads per WebSocket to eliminate asyncio contention.

        Args:
            duration: Duration of the evaluation in seconds
        """
        if not self.bridge_ready:
            raise RuntimeError("[RUN SCENARIO] Bridge is not ready, please call `bridge.prepare_for_scenario()` first")
        if self.needs_reset:
            raise RuntimeError(
                "Bridge needs reset before running a new scenario, please call `bridge.reset()` or `bridge.prepare_for_scenario()` first"
            )

        logger.info(f"[RUN SCENARIO] Running scenario for {duration} seconds...")
        self.metrics.start_time = datetime.now()
        self.metrics.end_time = None
        # Clear state for this run
        self.stop_event.clear()
        self.stop_reason = STOP_REASON_TIMEOUT
        self.sent_to_agent_chunks = []
        self.sent_to_user_chunks = []
        self.user_context_history = None
        self.agent_context_history = None

        # Clear thread-safe queues
        self.user_to_agent_queue = queue.Queue()
        self.agent_to_user_queue = queue.Queue()

        # Create and start threads
        user_thread = threading.Thread(target=self.user_websocket_thread, args=(duration,), name="UserWebSocketThread")
        agent_thread = threading.Thread(
            target=self.agent_websocket_thread, args=(duration,), name="AgentWebSocketThread"
        )

        # Start both threads
        logger.info("[RUN SCENARIO] Starting threads for user and agent...")

        # Set thread start timestamp for conversation log timing (aligns with bridge_audio_log.wav)
        loop = asyncio.get_event_loop()
        self.metrics.thread_start_timestamp = loop.time()

        user_thread.start()
        agent_thread.start()

        # Wait for both threads to complete (in async context)
        logger.info("[RUN SCENARIO] Waiting for threads to complete...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, user_thread.join)
        await loop.run_in_executor(None, agent_thread.join)

        logger.info("[RUN SCENARIO] Both user and agent threads completed")
        self.metrics.end_time = datetime.now()
        # Finalize any in-progress turns at end of scenario
        loop = asyncio.get_event_loop()
        timestamp = loop.time()
        self._finalize_speaker_turn("user", timestamp)
        self._finalize_speaker_turn("agent", timestamp)

        # Write conversation log with post-hoc latency calculation
        self._save_final_response()
        self._save_conversation_log()
        self._save_audio_log()
        self._save_seglst()
        self._save_user_agent_history()
        logger.info(f"[RUN SCENARIO] Saved audio and logs to: {Path(self.log_file).parent}")
        self.needs_reset = True
        self.bridge_ready = False

    def _build_conversation_log(self):
        """
        Build conversation log entries from finalized segments with computed latencies.

        Called after all segments are finalized so that latency calculation has access
        to all user and agent segments. For each agent segment, latency is computed as:
            agent.start_time - previous_user.end_time
        Positive = normal response delay, negative = agent interrupted/barged in early.

        Applies turn_start_offset_secs and turn_end_offset_secs to match seglst timestamps.
        """
        sorted_segments = sorted(self.metrics.segments, key=lambda s: s.start_time)

        self.metrics.log_entries = []
        last_user_end = None

        for seg in sorted_segments:
            start = seg.start_time + self.turn_start_offset_secs
            end = seg.end_time + self.turn_end_offset_secs
            # Ensure offsets don't produce negative duration
            if end <= start:
                start = seg.start_time
                end = seg.end_time

            if seg.speaker == "user":
                last_user_end = end
                latency_ms = None
            else:  # agent
                if last_user_end is not None:
                    latency_ms = (start - last_user_end) * 1000
                else:
                    latency_ms = None

            log_entry = self._format_turn_log(seg.speaker, seg.transcript, start, end, latency_ms)
            self.metrics.log_entries.append((start, log_entry))

    def _save_conversation_log(self):
        """Build and write conversation log entries sorted by start time, with computed latencies."""
        if not self.log_file or not self.metrics.segments:
            return

        # Build log entries from finalized segments with post-hoc latency
        self._build_conversation_log()

        try:
            sorted_entries = sorted(self.metrics.log_entries, key=lambda x: x[0])
            with open(self.log_file, "a") as f:
                for _start_time, log_entry in sorted_entries:
                    f.write(log_entry)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write(f"End time: {self.metrics.end_time.isoformat()}\n")
                f.write(f"Stop reason: {self.stop_reason}\n")
                f.write("=" * 80 + "\n")
            logger.info(f"[LOG] Wrote {len(sorted_entries)} conversation turns to log file (sorted by time)")
        except Exception as e:
            logger.error(f"[LOG] Error writing sorted log entries: {e}")

    @staticmethod
    def _resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio array using soxr. Returns int16 array."""
        if from_rate == to_rate or len(audio) == 0:
            return audio
        return soxr.resample(audio, from_rate, to_rate, quality="VHQ").astype(np.int16)

    def _save_audio_log(self):
        """Save final sent audio chunks to disk as stereo WAV for debugging."""
        if not self.bridge_audio_file:
            logger.warning("[DEBUG] No bridge_audio_file to save audio")
            return

        output_path = Path(self.bridge_audio_file)

        if not self.sent_to_agent_chunks and not self.sent_to_user_chunks:
            logger.info("[DEBUG] No audio chunks to save")
            return

        logger.info(f"[DEBUG] Saving bridge audio log to {output_path}")
        # Convert audio chunks to numpy arrays
        # Channel 0 (Left): USER→AGENT audio at agent_input_sample_rate
        # Channel 1 (Right): AGENT→USER audio at user_input_sample_rate

        channel0 = np.array([], dtype=np.int16)
        channel1 = np.array([], dtype=np.int16)

        if self.sent_to_agent_chunks:
            audio_data = b"".join(self.sent_to_agent_chunks)
            channel0 = np.frombuffer(audio_data, dtype=np.int16)

        if self.sent_to_user_chunks:
            audio_data = b"".join(self.sent_to_user_chunks)
            channel1 = np.frombuffer(audio_data, dtype=np.int16)

        # Resample both channels to output_sample_rate (typically 16kHz)
        target_rate = self.output_sample_rate

        channel0 = self._resample_audio(channel0, self.agent_input_sample_rate, target_rate)
        channel1 = self._resample_audio(channel1, self.user_input_sample_rate, target_rate)

        # Pad shorter channel with silence to match longer one
        max_length = max(len(channel0), len(channel1))

        if len(channel0) < max_length:
            channel0 = np.pad(channel0, (0, max_length - len(channel0)), mode='constant', constant_values=0)

        if len(channel1) < max_length:
            channel1 = np.pad(channel1, (0, max_length - len(channel1)), mode='constant', constant_values=0)

        # Interleave channels for stereo: [L, R, L, R, ...]
        stereo_data = np.empty(max_length * 2, dtype=np.int16)
        stereo_data[0::2] = channel0  # Left channel (USER→AGENT)
        stereo_data[1::2] = channel1  # Right channel (AGENT→USER)

        # Save as stereo WAV
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(target_rate)
            wav_file.writeframes(stereo_data.tobytes())

        duration = max_length / target_rate
        logger.info(f"[DEBUG] Saved stereo bridge audio: {output_path}")
        logger.info(f"        Left (USER→AGENT): {len(self.sent_to_agent_chunks)} chunks")
        logger.info(f"        Right (AGENT→USER): {len(self.sent_to_user_chunks)} chunks")
        logger.info(f"        Duration: {duration:.2f}s, Sample rate: {target_rate}Hz")

    def _save_bot_server_history(self, output_dir: Union[str, Path], context_history: dict):
        """Save the bot server context history to a JSON file under the output directory."""
        if not output_dir:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        context = eval(context_history.get("context", "[]"))
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except Exception as e:
                logger.error(f"Error loading context into json object: {e}. Context: {context}")
        else:
            context = context
        logs = context_history.get("logs", "")
        context_file = output_dir / "context.json"
        log_file = output_dir / "logs.txt"
        with open(context_file, "w") as f:
            json.dump(context, f, indent=2)
        with open(log_file, "w") as f:
            f.write(logs)

    def _save_user_agent_history(self):
        """Save the user and agent context history to a JSON file under the output directory."""
        if not self.output_dir:
            return

        output_dir_user = Path(self.output_dir) / "bot_logs_user"
        output_dir_agent = Path(self.output_dir) / "bot_logs_agent"
        self._save_bot_server_history(output_dir_user, self.user_context_history)
        self._save_bot_server_history(output_dir_agent, self.agent_context_history)

    async def _monitor_user_message(self, frame):
        """
        Monitor user messages for timing and transcripts.

        Turn lifecycle: BOT_STARTED_SPEAKING → BOT_TTS_TEXT (accumulate) → BOT_STOPPED_SPEAKING (finalize).
        """
        timestamp = asyncio.get_event_loop().time()

        if frame is None:
            return

        # logger.debug(f"[USER MONITOR] Frame type: {type(frame).__name__}, has audio: {hasattr(frame, 'audio')}")

        # Handle audio frames
        if hasattr(frame, 'audio') and frame.audio:
            self.metrics.user_last_audio_time = timestamp
            return

        # Handle RTVI protocol messages
        if not (hasattr(frame, 'message') and frame.message):
            return

        data = json.loads(frame.message) if isinstance(frame.message, str) else frame.message
        message_type = data.get("type", "")

        if message_type == RTVI_BOT_STARTED_SPEAKING:
            # Defensive: close previous turn if it wasn't properly stopped
            self._finalize_speaker_turn("user", timestamp)

            # Start new turn
            relative_time = self._get_relative_time(timestamp)
            self.metrics.current_user_segment = SegmentEntry(
                start_time=relative_time, end_time=relative_time, speaker="user", transcript=""
            )
            self.metrics.user_current_transcript = ""
            logger.debug("[TIMING] User started speaking")

        elif message_type == RTVI_BOT_TTS_TEXT:
            text = str(data.get("data", {}).get("text", ""))
            logger.debug(f"[USER TTS] {text}")
            if text:
                self.metrics.user_current_transcript += text

        elif message_type == RTVI_BOT_STOPPED_SPEAKING:
            logger.debug("[USER STOPPED SPEAKING]")
            self.metrics.user_last_audio_time = timestamp
            self.metrics.waiting_for_agent_response = True

            segment = self._finalize_speaker_turn("user", timestamp)
            if segment:
                self.metrics.last_user_transcript = segment.transcript
                self.metrics.turns.append(
                    {"timestamp": datetime.now().isoformat(), "role": "user", "text": segment.transcript}
                )

    async def _monitor_agent_message(self, frame):
        """
        Monitor agent messages for timing and transcripts.

        Turn lifecycle: BOT_STARTED_SPEAKING → BOT_TTS_TEXT (accumulate) → BOT_STOPPED_SPEAKING (finalize).
        Latency is measured from user's last audio to agent's first audio frame.
        """
        timestamp = asyncio.get_event_loop().time()

        if frame is None:
            return

        # logger.debug(f"[AGENT MONITOR] Frame type: {type(frame).__name__}, has audio: {hasattr(frame, 'audio')}")

        # Handle audio frames — measure latency on first agent audio after user stops
        if hasattr(frame, 'audio') and frame.audio:
            if self.metrics.waiting_for_agent_response and self.metrics.user_last_audio_time:
                latency_ms = (timestamp - self.metrics.user_last_audio_time) * 1000
                latency = ResponseLatency(
                    user_stop_time=self.metrics.user_last_audio_time,
                    agent_start_time=timestamp,
                    latency_ms=latency_ms,
                    user_transcript=self.metrics.last_user_transcript,
                )
                self.metrics.latencies.append(latency)
                self.metrics.waiting_for_agent_response = False
                logger.info(f"[LATENCY] Response latency: {latency_ms:.1f}ms")

            self.metrics.agent_last_audio_time = timestamp
            return

        # Handle RTVI protocol messages
        if not (hasattr(frame, 'message') and frame.message):
            return

        data = json.loads(frame.message) if isinstance(frame.message, str) else frame.message
        message_type = data.get("type", "")

        if message_type == RTVI_BOT_STARTED_SPEAKING:
            logger.debug("[AGENT STARTED SPEAKING]")
            # Defensive: close previous turn if it wasn't properly stopped
            self._finalize_speaker_turn("agent", timestamp)

            # Start new turn
            relative_time = self._get_relative_time(timestamp)
            self.metrics.current_agent_segment = SegmentEntry(
                start_time=relative_time, end_time=relative_time, speaker="agent", transcript=""
            )
            self.metrics.agent_current_transcript = ""

        elif message_type == RTVI_BOT_TTS_TEXT:
            text = str(data.get("data", {}).get("text", ""))
            logger.debug(f"[AGENT TTS] {text}")
            if text:
                self.metrics.agent_current_transcript += text

        elif message_type == RTVI_BOT_STOPPED_SPEAKING:
            logger.debug("[AGENT STOPPED SPEAKING]")
            segment = self._finalize_speaker_turn("agent", timestamp)
            if segment:
                # Update the last latency measurement with agent transcript
                if self.metrics.latencies and not self.metrics.latencies[-1].agent_transcript:
                    self.metrics.latencies[-1].agent_transcript = segment.transcript

                self.metrics.turns.append(
                    {"timestamp": datetime.now().isoformat(), "role": "agent", "text": segment.transcript}
                )

        elif message_type == RTVI_BOT_SERVER_MESSAGE:
            text = str(data.get("data", {}).get("text", ""))
            if text:
                logger.info(f"[AGENT SERVER MESSAGE] {text}")
                if text.startswith(FINAL_RESPONSE_START_TAG) and text.endswith(FINAL_RESPONSE_END_TAG):
                    final_response = text[len(FINAL_RESPONSE_START_TAG) : -len(FINAL_RESPONSE_END_TAG)]
                    logger.info(f"[AGENT FINAL RESPONSE] {final_response}")
                    self.metrics.agent_final_response.append(final_response)
                    self.metrics.agent_final_response_time.append(timestamp)
                    logger.info("[AGENT] Final response saved")
                if text.startswith(EXIT_MESSAGE_START_TAG) and text.endswith(EXIT_MESSAGE_END_TAG):
                    exit_message = text[len(EXIT_MESSAGE_START_TAG) : -len(EXIT_MESSAGE_END_TAG)]
                    logger.info(f"[AGENT] Exit message received, signaling early stop. Exit message: {exit_message}")
                    self.stop_reason = STOP_REASON_EXIT
                    self.stop_event.set()
                    self.metrics.end_time = datetime.now()

    def _save_final_response(self):
        """Save the agent's final response to a JSON file under the output directory."""
        if not self.output_dir:
            return

        results = []
        for final_response in self.metrics.agent_final_response:
            try:
                response_obj = json.loads(final_response)
            except (json.JSONDecodeError, TypeError):
                response_obj = {"message": final_response}
            results.append(response_obj)

        output_path = Path(self.output_dir) / self.final_response_file
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Final agent response saved: {output_path}")
        except Exception as e:
            logger.error(f"Error saving final agent response: {e}")

    def _save_seglst(self):
        """Save segLST transcript file with offset-adjusted timestamps."""
        if not self.seglst_file or not self.metrics.segments:
            return

        try:
            session_id = self.scenario_name or "evaluation"
            segments_json = []
            sorted_segments = sorted(self.metrics.segments, key=lambda s: s.start_time)

            for seg in sorted_segments:
                start = seg.start_time + self.turn_start_offset_secs
                end = seg.end_time + self.turn_end_offset_secs
                if end <= start:
                    start = seg.start_time
                    end = seg.end_time

                segments_json.append(
                    {
                        "session_id": session_id,
                        "words": seg.transcript,
                        "speaker": seg.speaker,
                        "start_time": start,
                        "end_time": end,
                    }
                )

            with open(self.seglst_file, 'w') as f:
                json.dump(segments_json, f, indent=2)

            logger.info(f"segLST saved: {self.seglst_file} ({len(sorted_segments)} segments)")

        except Exception as e:
            logger.error(f"Error saving segLST: {e}")
            import traceback

            traceback.print_exc()

    async def disconnect(self, print_stats: bool = False):
        """
        Disconnect from both user and agent.

        Args:
            print_stats: If True, print final latency statistics (default: True)
                        Set to False when disconnecting during scenario resets
        """
        if print_stats:
            self.metrics.end_time = datetime.now()

        if self.user_ws:
            await self.user_ws.close()
        if self.agent_ws:
            await self.agent_ws.close()

        logger.info("Disconnected from user and agent")

        # Log final statistics only if requested
        if print_stats:
            latency_stats = self.metrics.get_latency_stats()
            if latency_stats['count'] > 0:
                logger.info(f"\nFinal Latency Statistics:")
                logger.info(f"  Measurements: {latency_stats['count']}")
                logger.info(f"  Mean: {latency_stats['mean_ms']:.1f}ms")
                logger.info(f"  P50: {latency_stats['p50_ms']:.1f}ms")
                logger.info(f"  P95: {latency_stats['p95_ms']:.1f}ms")
                logger.info(f"  Min: {latency_stats['min_ms']:.1f}ms")
                logger.info(f"  Max: {latency_stats['max_ms']:.1f}ms")

    def get_metrics(self):
        """Get evaluation metrics"""
        duration = 0
        if self.metrics.start_time and self.metrics.end_time:
            duration = (self.metrics.end_time - self.metrics.start_time).total_seconds()

        latency_stats = self.metrics.get_latency_stats()

        return {
            "total_turns": len(self.metrics.turns),
            "duration_seconds": duration,
            "turns": self.metrics.turns,
            "latency_stats": latency_stats,
            "latencies": [
                {
                    "user_transcript": l.user_transcript,
                    "agent_transcript": l.agent_transcript,
                    "latency_ms": l.latency_ms,
                }
                for l in self.metrics.latencies
            ],
        }
