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

"""
RTVI Evaluation Bridge

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
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import websockets
from loguru import logger
from pipecat.frames.frames import Frame, OutputAudioRawFrame
from pipecat.processors.frameworks.rtvi import (
    RTVIBotStartedSpeakingMessage,
    RTVIBotStoppedSpeakingMessage,
    RTVIBotTranscriptionMessage,
    RTVITextMessageData,
)
from pipecat.serializers.protobuf import MessageFrame, ProtobufFrameSerializer

# Import AudioStream for buffering and resampling
from nemo.agents.voice_agent.utils.audio import AudioStream

# RTVI message type constants - automatically adapts to pipecat changes
RTVI_BOT_STOPPED_SPEAKING = RTVIBotStoppedSpeakingMessage().type
RTVI_BOT_STARTED_SPEAKING = RTVIBotStartedSpeakingMessage().type
RTVI_BOT_TRANSCRIPTION = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text="")).type


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

    # Audio timing state
    user_last_audio_time: Optional[float] = None
    agent_last_audio_time: Optional[float] = None
    waiting_for_agent_response: bool = False
    last_user_transcript: str = ""

    # Transcript accumulation (segments arrive incrementally)
    user_current_transcript: str = ""
    agent_current_transcript: str = ""

    # Audio recording (for stereo WAV output)
    user_audio_chunks: List[bytes] = field(default_factory=list)
    agent_audio_chunks: List[bytes] = field(default_factory=list)
    audio_sample_rate: int = 16000  # Default sample rate
    audio_start_timestamp: Optional[float] = None

    # Segment tracking for segLST output
    segments: List[SegmentEntry] = field(default_factory=list)
    current_segment_start: Optional[float] = None

    def get_latency_stats(self):
        """Calculate latency statistics"""
        if not self.latencies:
            return {
                "count": 0,
                "mean_ms": 0,
                "median_ms": 0,
                "p95_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
            }

        latencies_sorted = sorted([l.latency_ms for l in self.latencies])
        count = len(latencies_sorted)

        return {
            "count": count,
            "mean_ms": sum(latencies_sorted) / count,
            "median_ms": latencies_sorted[count // 2],
            "p95_ms": latencies_sorted[int(count * 0.95)] if count > 0 else 0,
            "min_ms": latencies_sorted[0],
            "max_ms": latencies_sorted[-1],
        }


class RTVIEvaluationBridge:
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
        log_file: Optional[str] = None,
        audio_file: Optional[str] = None,
        user_output_sample_rate: int = 24000,
        agent_output_sample_rate: int = 24000,
        user_input_sample_rate: int = 16000,
        agent_input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
        audio_chunk_in_seconds: float = 0.016,
    ):
        self.user_url = user_url
        self.agent_url = agent_url
        self.log_file = log_file
        self.audio_file = audio_file
        self.user_output_sample_rate = user_output_sample_rate
        self.agent_output_sample_rate = agent_output_sample_rate
        self.user_input_sample_rate = user_input_sample_rate
        self.agent_input_sample_rate = agent_input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.audio_chunk_in_seconds = audio_chunk_in_seconds

        # Random burst mode configuration (simulates browser's irregular sending pattern)
        self.use_burst_mode = False
        self.burst_size_range = (2, 4)  # Random 2-4 frames per burst
        self.burst_delay_ms = 1  # 1ms between frames in burst
        # Pause calculated per burst: (burst_size × 16ms) - burst_duration
        # This maintains 16ms average per frame while varying the pattern

        # Grace period and timeout configuration for send loops
        self.grace_period = 5.0  # Extra time to drain audio after main duration
        self.no_audio_timeout = 2.0  # Stop if no audio for N seconds during grace period
        self.max_consecutive_silence = 5  # Stop after N consecutive silence chunks in grace period

        self.user_ws = None
        self.agent_ws = None

        self.metrics = EvaluationMetrics()
        self.metrics.audio_sample_rate = output_sample_rate

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
            logger.info(f"  Example patterns:")
            logger.info(f"    2 frames: 1ms burst + 31ms pause = 32ms (16ms avg)")
            logger.info(f"    3 frames: 2ms burst + 46ms pause = 48ms (16ms avg)")
            logger.info(f"    4 frames: 3ms burst + 61ms pause = 64ms (16ms avg)")
        else:
            logger.info(f"Steady mode: sending at constant {self.audio_chunk_in_seconds * 1000:.0f}ms intervals")

        # Initialize log file
        self.init_log_file(log_file)

    def init_log_file(self, log_file: str = None):
        """Initialize the log file"""
        if log_file:
            self.log_file = log_file
        if not self.log_file:
            return False
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

        self.metrics.start_time = datetime.now()

        # Send RTVI client-ready handshake to both agents
        await self._send_client_ready(self.user_ws)
        await self._send_client_ready(self.agent_ws)

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

    async def update_user_prompt(self, new_prompt: str, auto_reset: bool = False, add_suffix: bool = True):
        """
        Update user's system prompt via RTVI action.

        Args:
            new_prompt: New system prompt text
            auto_reset: If True, also sends reset action after updating prompt
            add_suffix: If True, add previously configured system prompt suffix to the new prompt
        """
        logger.info(f"Updating user prompt: {new_prompt[:100]}...")

        # Create RTVI action message
        action_msg = {
            "label": "rtvi-ai",
            "type": "action",
            "id": f"update_prompt_{datetime.now().timestamp()}",
            "data": {
                "service": "context",
                "action": "update_system_prompt",
                "arguments": [{"name": "prompt", "value": new_prompt}, {"name": "add_suffix", "value": add_suffix}],
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

    async def update_agent_prompt(self, new_prompt: str, auto_reset: bool = True, add_suffix: bool = True):
        """
        Update agent's system prompt via RTVI action.

        Args:
            new_prompt: New system prompt text
            auto_reset: If True, also sends reset action after updating prompt
            add_suffix: If True, add previously configured system prompt suffix to the new prompt
        """
        logger.info(f"Updating agent prompt: {new_prompt[:100]}...")

        # Create RTVI action message
        action_msg = {
            "label": "rtvi-ai",
            "type": "action",
            "id": f"update_prompt_{datetime.now().timestamp()}",
            "data": {
                "service": "context",
                "action": "update_system_prompt",
                "arguments": [{"name": "prompt", "value": new_prompt}, {"name": "add_suffix", "value": add_suffix}],
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

    async def reset_conversation(self):
        """
        Reset both agents' conversation history.
        Useful to clear context between evaluation scenarios.
        """
        logger.info("Resetting both agents...")
        await self._send_reset_action(self.user_ws, "user")
        await self._send_reset_action(self.agent_ws, "agent")

        # Reset latency tracking state
        self.metrics.user_last_audio_time = None
        self.metrics.agent_last_audio_time = None
        self.metrics.waiting_for_agent_response = False
        self.metrics.last_user_transcript = ""

        # Clear accumulated transcript segments
        self.metrics.user_current_transcript = ""
        self.metrics.agent_current_transcript = ""

        logger.info("Both agents reset complete")

    async def reset_agent(self):
        """
        Reset agent's conversation history.
        Useful to clear context between evaluation scenarios.
        """
        logger.info("Resetting agent...")
        await self._send_reset_action(self.agent_ws, "agent")

        logger.info("Agent reset complete")

    async def reset_user(self):
        """
        Reset user's conversation history.
        Useful to clear context between evaluation scenarios.
        """
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

    def _should_forward_frame(self, frame, source="unknown") -> bool:
        """
        Determine if a frame should be forwarded between agents.
        Only forwards audio frames - filters out all other messages.

        Args:
            frame: Already deserialized frame from receive loop
            source: Source of the frame for logging
        """
        try:
            if frame is None:
                return False  # Don't forward unknown frames

            # Only forward raw audio frames (not message frames)
            if hasattr(frame, 'audio') and frame.audio:
                return True

            # Check if it's a message frame with audio data
            if hasattr(frame, 'message') and frame.message:
                if isinstance(frame.message, str):
                    data = json.loads(frame.message)
                else:
                    data = frame.message

                msg_type = data.get("type", "")

                # Only forward raw audio messages
                if msg_type in ["raw-audio", "raw-audio-batch"]:
                    return True
                else:
                    # Filter out all other messages
                    return False

            return False  # Don't forward anything else
        except Exception as e:
            logger.debug(f"Error checking frame for forwarding: {e}")
            return False  # Don't forward if we can't determine

    async def _reconnect_websocket(self, direction: str, max_retries: int = 3) -> websockets.WebSocketClientProtocol:
        """
        Reconnect to a websocket endpoint.

        Args:
            direction: "USER" or "AGENT"
            max_retries: Maximum reconnection attempts

        Returns:
            New WebSocket connection
        """
        url = self.user_url if direction == "USER" else self.agent_url

        for attempt in range(max_retries):
            try:
                logger.warning(
                    f"[RECONNECT {direction}] Attempting reconnection to {url} (attempt {attempt + 1}/{max_retries})"
                )
                ws = await websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=10)

                # Send client-ready handshake
                await self._send_client_ready(ws)

                # Update the connection reference
                if direction == "USER":
                    self.user_ws = ws
                else:
                    self.agent_ws = ws

                logger.info(f"[RECONNECT {direction}] Successfully reconnected")
                return ws

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"[RECONNECT {direction}] Failed (attempt {attempt + 1}): {e}, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[RECONNECT {direction}] Failed after {max_retries} attempts")
                    raise

    async def _receive_and_queue_audio(
        self,
        source_ws: websockets.WebSocketClientProtocol,
        audio_stream: AudioStream,
        monitor_func: Callable[[Frame], None],
        direction: str,
        auto_reconnect: bool = True,
    ):
        """
        Receive audio from websocket and put into AudioStream. Automatically reconnects on disconnection.

        Args:
            source_ws: Source websocket to receive from
            audio_stream: AudioStream to store and resample audio chunks
            monitor_func: Monitoring function for metrics
            direction: For logging (e.g., "USER", "AGENT")
            auto_reconnect: Whether to automatically reconnect on disconnection
        """
        logger.info(f"[RECEIVE {direction}] Loop started")
        ws = source_ws
        prev_audio_time = None
        import time

        while True:
            try:
                async for message in ws:
                    # Deserialize ONCE for all processing
                    try:
                        frame = await self.serializer.deserialize(message)
                        if frame is None:
                            continue
                    except Exception as e:
                        logger.error(f"[RECEIVE {direction}] Error deserializing message: {e}", exc_info=True)
                        continue

                    # Monitor for metrics and transcripts (pass deserialized frame)
                    await monitor_func(frame)

                    # Check if this is an audio frame (pass deserialized frame)
                    should_forward = self._should_forward_frame(frame, source=direction)
                    if not should_forward:
                        continue

                    # Put audio into AudioStream (auto-resamples)
                    try:
                        if hasattr(frame, 'audio') and frame.audio:
                            await audio_stream.put(frame.audio)
                            current_audio_time = time.time()
                            if prev_audio_time is None:
                                time_since_last_audio = 0
                                prev_audio_time = current_audio_time
                            else:
                                time_since_last_audio = current_audio_time - prev_audio_time
                                prev_audio_time = current_audio_time
                            logger.debug(
                                f"[RECEIVE {direction}] Put {len(frame.audio)} bytes into AudioStream, time since last audio: {time_since_last_audio:.6f}s"
                            )
                    except Exception as e:
                        logger.error(f"[RECEIVE {direction}] Error processing audio: {e}", exc_info=True)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"[RECEIVE {direction}] Connection closed: {e}")
                if auto_reconnect:
                    try:
                        ws = await self._reconnect_websocket(direction)
                        logger.info(f"[RECEIVE {direction}] Reconnected, resuming receive loop")
                        continue  # Continue receiving with new connection
                    except Exception as reconnect_error:
                        logger.error(f"[RECEIVE {direction}] Reconnection failed: {reconnect_error}")
                        break
                else:
                    break

            except asyncio.CancelledError:
                logger.info(f"[RECEIVE {direction}] Task cancelled")
                break

            except Exception as e:
                logger.error(f"[RECEIVE {direction}] Unexpected error: {e}", exc_info=True)
                if auto_reconnect:
                    logger.info(f"[RECEIVE {direction}] Attempting to recover...")
                    await asyncio.sleep(1)
                    try:
                        ws = await self._reconnect_websocket(direction)
                        continue
                    except:
                        break
                else:
                    break

        logger.info(f"[RECEIVE {direction}] Loop exited")

    async def _send_audio_stream(
        self,
        audio_stream: AudioStream,
        dest_ws: websockets.WebSocketClientProtocol,
        direction: str,
        auto_reconnect: bool = True,
    ):
        """
        Send audio stream at fixed intervals from AudioStream.

        Args:
            audio_stream: AudioStream containing buffered and resampled audio
            dest_ws: Destination websocket to send to
            direction: For logging (e.g., "USER→AGENT", "AGENT→USER")
            auto_reconnect: Whether to automatically reconnect on disconnection
        """
        # Determine websocket direction for reconnection
        ws_direction = None
        if direction == "AGENT→USER":
            ws_direction = "USER"
        elif direction == "USER→AGENT":
            ws_direction = "AGENT"

        logger.info(f"[SEND {direction}] Loop started (interval: {self.audio_chunk_in_seconds}s)")

        ws = dest_ws
        consecutive_errors = 0
        max_consecutive_errors = 5

        # Time-based scheduling to avoid drift from processing time
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        chunk_count = 0

        try:
            while True:
                # Calculate when next chunk should be sent (absolute time)
                chunk_count += 1
                next_send_time = start_time + (chunk_count * self.audio_chunk_in_seconds)
                current_time = loop.time()

                # Calculate how long to wait for next chunk
                wait_duration = max(0.001, next_send_time - current_time)  # Minimum 1ms

                if wait_duration < 0:
                    logger.warning(f"[SEND {direction}][{chunk_count}] Behind schedule by {-wait_duration:.6f}s")
                    wait_duration = 0.001  # Minimum wait to check for audio

                # Get audio from AudioStream with no wait
                # This only tries to read the audio cache once, and returns silence
                # immediately if no audio is available.
                # If audio arrives early, we still sleep the remaining time to maintain cadence
                wait_start = loop.time()
                audio_to_send = await audio_stream.get_nowait()
                wait_elapsed = loop.time() - wait_start

                # If audio arrived early, sleep the remaining time to maintain 16ms cadence
                remaining_sleep = wait_duration - wait_elapsed
                if remaining_sleep > 0.001:  # More than 1ms remaining
                    logger.debug(
                        f"[SEND {direction}][{chunk_count}] Audio arrived early, sleeping {remaining_sleep:.6f}s more"
                    )
                    await asyncio.sleep(remaining_sleep)
                audio_len_in_seconds = len(audio_to_send) / 2 / audio_stream.output_sample_rate
                logger.debug(
                    f"[SEND {direction}][{chunk_count}] Got {len(audio_to_send)} bytes from AudioStream of {audio_len_in_seconds:.4f} seconds"
                )

                # Debug: accumulate sent chunks for analysis
                if direction == "USER→AGENT":
                    self.sent_to_agent_chunks.append(audio_to_send)
                elif direction == "AGENT→USER":
                    self.sent_to_user_chunks.append(audio_to_send)

                # Create output frame and send
                try:
                    output_frame = OutputAudioRawFrame(
                        audio=audio_to_send, sample_rate=audio_stream.output_sample_rate, num_channels=1
                    )
                    serialized = await self.serializer.serialize(output_frame)
                    await ws.send(serialized)

                    consecutive_errors = 0  # Reset error count on success

                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"[SEND {direction}] Connection closed: {e}")
                    if auto_reconnect and ws_direction:
                        try:
                            ws = await self._reconnect_websocket(ws_direction)
                            logger.info(f"[SEND {direction}] Reconnected, resuming send loop")
                            consecutive_errors = 0
                        except Exception as reconnect_error:
                            logger.error(f"[SEND {direction}] Reconnection failed: {reconnect_error}")
                            break
                    else:
                        break

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        f"[SEND {direction}] Error sending audio (consecutive errors: {consecutive_errors}): {e}"
                    )
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"[SEND {direction}] Too many consecutive errors ({consecutive_errors}), stopping"
                        )
                        break
                    if auto_reconnect and ws_direction:
                        try:
                            ws = await self._reconnect_websocket(ws_direction)
                            consecutive_errors = 0
                        except:
                            pass

        except asyncio.CancelledError:
            logger.info(f"[SEND {direction}] Task cancelled")
        except Exception as e:
            logger.error(f"[SEND {direction}] Fatal error: {e}")
        finally:
            logger.info(f"[SEND {direction}] Loop exited")

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
                                self._receive_user_to_queue(user_ws),
                                # Get raw audio from queue, send to user (handles its own timeout)
                                self._send_agent_to_user(user_ws, agent_to_user_stream, duration),
                            ),
                            timeout=duration + self.grace_period + 1.0,  # Extra 1s buffer for cleanup
                        )
                    except asyncio.TimeoutError:
                        logger.info("[USER THREAD] Overall timeout reached, stopping receive loop")

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
                                self._receive_agent_to_queue(agent_ws),
                                send_kickoff(),
                            ),
                            timeout=duration + self.grace_period + 1.0,  # Extra 1s buffer for cleanup
                        )
                    except asyncio.TimeoutError:
                        logger.info("[AGENT THREAD] Overall timeout reached, stopping receive loop")

            except Exception as e:
                logger.error(f"[AGENT THREAD] Error: {e}", exc_info=True)
            finally:
                logger.info("[AGENT THREAD] Exiting")

        try:
            loop.run_until_complete(agent_loop())
        finally:
            loop.close()

    async def _receive_user_to_queue(self, user_ws):
        """Receive audio from user WebSocket and put into queue for agent thread."""
        logger.info("[USER→AGENT] Starting receive loop")

        try:
            async for message in user_ws:
                # Deserialize frame
                try:
                    frame = await self.serializer.deserialize(message)
                    if frame is None:
                        continue
                except Exception as e:
                    logger.error(f"[USER→AGENT] Deserialization error: {e}")
                    continue

                # Monitor user messages
                await self._monitor_user_message(frame)

                # Check if this is audio
                if hasattr(frame, 'audio') and frame.audio:
                    # Put raw audio into thread-safe queue for agent thread
                    self.user_to_agent_queue.put(frame.audio)
                    logger.debug(f"[USER→AGENT] Queued {len(frame.audio)} bytes of user audio")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[USER→AGENT] Receive WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"[USER→AGENT] Receive error: {e}", exc_info=True)

    async def _send_agent_to_user(self, user_ws, audio_stream: AudioStream, duration: int):
        """Get audio from queue, process through AudioStream, send to user WebSocket."""
        logger.info("[AGENT→USER] Starting send loop")

        loop = asyncio.get_event_loop()
        start_time = loop.time()
        chunk_count = 0
        consecutive_silence = 0  # Track consecutive silence chunks (local counter)

        try:
            last_audio_time = loop.time()
            queue_empty_for_drain = False  # Flag to indicate we're draining AudioStream buffer

            while True:
                current_time = loop.time()
                elapsed = current_time - start_time

                # Check if we're past the main duration
                in_grace_period = elapsed > duration

                # Stop conditions:
                # 1. Exceeded duration + grace period
                # 2. In grace period and no audio for no_audio_timeout seconds
                # 3. In grace period and consecutive silence chunks (buffer drained)
                if elapsed > (duration + self.grace_period):
                    logger.info(f"[AGENT→USER] Grace period expired after {elapsed:.1f}s, stopping")
                    break

                if in_grace_period and queue_empty_for_drain and consecutive_silence >= self.max_consecutive_silence:
                    logger.info(
                        f"[AGENT→USER] AudioStream buffer drained ({consecutive_silence} silence chunks), stopping"
                    )
                    break

                if in_grace_period and (current_time - last_audio_time) > self.no_audio_timeout:
                    logger.info(f"[AGENT→USER] No audio for {self.no_audio_timeout}s in grace period, stopping")
                    break

                # Empty all available agent audio from thread-safe queue (non-blocking)
                # This prevents stale audio buildup during burst pauses or LLM/TTS blocking
                chunks_retrieved = 0
                while True:
                    try:
                        agent_audio = self.agent_to_user_queue.get_nowait()
                        # Put into AudioStream for buffering/resampling
                        await audio_stream.put(agent_audio)
                        last_audio_time = current_time
                        chunks_retrieved += 1
                    except queue.Empty:
                        # Queue is empty, but we might still have audio in AudioStream buffer
                        if in_grace_period and chunks_retrieved == 0:
                            queue_empty_for_drain = True
                        break

                logger.debug(f"[AGENT→USER] Retrieved {chunks_retrieved} chunks from queue")

                # Burst sending: send N frames rapidly, then pause
                # Steady mode is just burst_size=1 (send 1 frame, pause 16ms, repeat)
                burst_size = (
                    random.randint(self.burst_size_range[0], self.burst_size_range[1]) if self.use_burst_mode else 1
                )

                # Send burst frames
                for frame_in_burst in range(burst_size):
                    if frame_in_burst > 0:
                        # Small delay between frames in burst
                        await asyncio.sleep(self.burst_delay_ms / 1000.0)

                    # Get audio from AudioStream
                    audio_to_send = await audio_stream.get_nowait()

                    # Check if it's silence
                    is_silence = audio_to_send == b'\x00' * len(audio_to_send)
                    if is_silence and queue_empty_for_drain:
                        consecutive_silence += 1
                    else:
                        consecutive_silence = 0

                    # Track sent audio
                    self.sent_to_user_chunks.append(audio_to_send)

                    # Create frame and send
                    output_frame = OutputAudioRawFrame(
                        audio=audio_to_send, sample_rate=audio_stream.output_sample_rate, num_channels=1
                    )
                    serialized = await self.serializer.serialize(output_frame)
                    await user_ws.send(serialized)

                    chunk_count += 1
                    if self.use_burst_mode:
                        logger.debug(
                            f"[AGENT→USER][{chunk_count}] Sent {len(audio_to_send)} bytes to user (burst {frame_in_burst+1}/{burst_size}, silence: {is_silence})"
                        )
                    else:
                        logger.debug(
                            f"[AGENT→USER][{chunk_count}] Sent {len(audio_to_send)} bytes to user (silence: {is_silence})"
                        )

                # Calculate pause to maintain average throughput
                total_cycle_time = burst_size * self.audio_chunk_in_seconds
                burst_duration = (burst_size - 1) * (self.burst_delay_ms / 1000.0)
                pause_duration = total_cycle_time - burst_duration

                if self.use_burst_mode:
                    logger.debug(
                        f"[AGENT→USER] Burst complete ({burst_size} frames), pausing {pause_duration*1000:.0f}ms"
                    )
                await asyncio.sleep(pause_duration)

            logger.info(f"[AGENT→USER] Send loop finished after {chunk_count} chunks")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[AGENT→USER] WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"[AGENT→USER] Send error: {e}", exc_info=True)

    async def _send_user_to_agent(self, agent_ws, audio_stream: AudioStream, duration: int):
        """Get audio from queue, process through AudioStream, send to agent WebSocket."""
        logger.info("[USER→AGENT] Starting send loop")

        loop = asyncio.get_event_loop()
        start_time = loop.time()
        chunk_count = 0
        consecutive_silence = 0  # Track consecutive silence chunks (local counter)

        try:
            last_audio_time = loop.time()
            queue_empty_for_drain = False  # Flag to indicate we're draining AudioStream buffer

            while True:
                current_time = loop.time()
                elapsed = current_time - start_time

                # Check if we're past the main duration
                in_grace_period = elapsed > duration

                # Stop conditions:
                # 1. Exceeded duration + grace period
                # 2. In grace period and no audio for no_audio_timeout seconds
                # 3. In grace period and consecutive silence chunks (buffer drained)
                if elapsed > (duration + self.grace_period):
                    logger.info(f"[USER→AGENT] Grace period expired after {elapsed:.1f}s, stopping")
                    break

                if in_grace_period and queue_empty_for_drain and consecutive_silence >= self.max_consecutive_silence:
                    logger.info(
                        f"[USER→AGENT] AudioStream buffer drained ({consecutive_silence} silence chunks), stopping"
                    )
                    break

                if in_grace_period and (current_time - last_audio_time) > self.no_audio_timeout:
                    logger.info(f"[USER→AGENT] No audio for {self.no_audio_timeout}s in grace period, stopping")
                    break

                # Empty all available user audio from thread-safe queue (non-blocking)
                # This prevents stale audio buildup during burst pauses or LLM/TTS blocking
                chunks_retrieved = 0
                while True:
                    try:
                        user_audio = self.user_to_agent_queue.get_nowait()
                        # Put into AudioStream for buffering/resampling
                        await audio_stream.put(user_audio)
                        last_audio_time = current_time
                        chunks_retrieved += 1
                    except queue.Empty:
                        # Queue is empty, but we might still have audio in AudioStream buffer
                        if in_grace_period and chunks_retrieved == 0:
                            queue_empty_for_drain = True
                        break

                logger.debug(f"[USER→AGENT] Retrieved {chunks_retrieved} chunks from queue")

                # Burst sending: send N frames rapidly, then pause
                # Steady mode is just burst_size=1 (send 1 frame, pause 16ms, repeat)
                burst_size = (
                    random.randint(self.burst_size_range[0], self.burst_size_range[1]) if self.use_burst_mode else 1
                )

                # Send burst frames
                for frame_in_burst in range(burst_size):
                    if frame_in_burst > 0:
                        # Small delay between frames in burst
                        await asyncio.sleep(self.burst_delay_ms / 1000.0)

                    # Get audio from AudioStream
                    audio_to_send = await audio_stream.get_nowait()

                    # Check if it's silence
                    is_silence = audio_to_send == b'\x00' * len(audio_to_send)
                    if is_silence and queue_empty_for_drain:
                        consecutive_silence += 1
                    else:
                        consecutive_silence = 0

                    # Track sent audio
                    self.sent_to_agent_chunks.append(audio_to_send)

                    # Create frame and send
                    output_frame = OutputAudioRawFrame(
                        audio=audio_to_send, sample_rate=audio_stream.output_sample_rate, num_channels=1
                    )
                    serialized = await self.serializer.serialize(output_frame)
                    await agent_ws.send(serialized)

                    chunk_count += 1
                    if self.use_burst_mode:
                        logger.debug(
                            f"[USER→AGENT][{chunk_count}] Sent {len(audio_to_send)} bytes to agent (burst {frame_in_burst+1}/{burst_size}, silence: {is_silence})"
                        )
                    else:
                        logger.debug(
                            f"[USER→AGENT][{chunk_count}] Sent {len(audio_to_send)} bytes to agent (silence: {is_silence})"
                        )

                # Calculate pause to maintain average throughput
                total_cycle_time = burst_size * self.audio_chunk_in_seconds
                burst_duration = (burst_size - 1) * (self.burst_delay_ms / 1000.0)
                pause_duration = total_cycle_time - burst_duration

                if self.use_burst_mode:
                    logger.debug(
                        f"[USER→AGENT] Burst complete ({burst_size} frames), pausing {pause_duration*1000:.0f}ms"
                    )
                await asyncio.sleep(pause_duration)

            logger.info(f"[USER→AGENT] Send loop finished after {chunk_count} chunks")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[USER→AGENT] WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"[USER→AGENT] Send error: {e}", exc_info=True)

    async def _receive_agent_to_queue(self, agent_ws):
        """Receive audio from agent WebSocket and put into queue for user thread."""
        logger.info("[AGENT→USER] Starting receive loop")

        try:
            async for message in agent_ws:
                # Deserialize frame
                try:
                    frame = await self.serializer.deserialize(message)
                    if frame is None:
                        continue
                except Exception as e:
                    logger.error(f"[AGENT→USER] Deserialization error: {e}")
                    continue

                # Monitor agent messages
                await self._monitor_agent_message(frame)

                # Check if this is audio
                if hasattr(frame, 'audio') and frame.audio:
                    # Put raw audio into thread-safe queue for user thread
                    self.agent_to_user_queue.put(frame.audio)
                    logger.debug(f"[AGENT→USER] Queued {len(frame.audio)} bytes of agent audio")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[AGENT→USER] Receive WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"[AGENT→USER] Receive error: {e}", exc_info=True)

    async def route_audio(self, duration: int = 300):
        """
        Route audio between agents and monitor conversation.
        Uses separate threads per WebSocket to eliminate asyncio contention.

        Args:
            duration: Duration of the evaluation in seconds
        """
        logger.info(f"[THREADED BRIDGE] Routing audio for {duration} seconds...")
        logger.info("[THREADED BRIDGE] Using 4-thread architecture for complete isolation")
        logger.info(f"  Thread 1: Receive from user WS")
        logger.info(f"  Thread 2: Send to user WS")
        logger.info(f"  Thread 3: Receive from agent WS")
        logger.info(f"  Thread 4: Send to agent WS")

        # Clear debug accumulation lists for this run (only final sent audio)
        self.sent_to_agent_chunks = []
        self.sent_to_user_chunks = []

        # Clear thread-safe queues
        while not self.user_to_agent_queue.empty():
            try:
                self.user_to_agent_queue.get_nowait()
            except queue.Empty:
                break
        while not self.agent_to_user_queue.empty():
            try:
                self.agent_to_user_queue.get_nowait()
            except queue.Empty:
                break

        logger.info(f"AudioStream configuration:")
        logger.info(f"  User→Agent: {self.user_output_sample_rate}Hz → {self.agent_input_sample_rate}Hz")
        logger.info(f"  Agent→User: {self.agent_output_sample_rate}Hz → {self.user_input_sample_rate}Hz")

        # Create and start threads
        user_thread = threading.Thread(target=self.user_websocket_thread, args=(duration,), name="UserWebSocketThread")
        agent_thread = threading.Thread(
            target=self.agent_websocket_thread, args=(duration,), name="AgentWebSocketThread"
        )

        # Start both threads
        logger.info("[THREADED BRIDGE] Starting threads...")
        user_thread.start()
        agent_thread.start()

        # Wait for both threads to complete (in async context)
        logger.info("[THREADED BRIDGE] Waiting for threads to complete...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, user_thread.join)
        await loop.run_in_executor(None, agent_thread.join)

        logger.info("[THREADED BRIDGE] Both threads completed")

        # Finalize any remaining transcripts and audio after routing ends
        logger.info("Finalizing any remaining transcripts and audio...")
        timestamp = time.time()  # Use time.time() since we're not in asyncio context

        # Finalize user transcript if any
        user_has_transcript = bool(self.metrics.user_current_transcript)
        if user_has_transcript:
            complete_text = self.metrics.user_current_transcript.strip()
            logger.info(f"[USER FINAL] {complete_text}")

            turn_data = {
                "timestamp": datetime.now().isoformat(),
                "role": "user",
                "text": complete_text,
            }
            self.metrics.turns.append(turn_data)

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] USER: {complete_text}\n")

            self.metrics.user_current_transcript = ""

        # Finalize agent transcript if any
        agent_has_transcript = bool(self.metrics.agent_current_transcript)
        if agent_has_transcript:
            complete_text = self.metrics.agent_current_transcript.strip()
            logger.info(f"[AGENT FINAL] {complete_text}")

            # Update the last latency measurement with complete agent transcript
            if self.metrics.latencies and not self.metrics.latencies[-1].agent_transcript:
                self.metrics.latencies[-1].agent_transcript = complete_text

            turn_data = {
                "timestamp": datetime.now().isoformat(),
                "role": "agent",
                "text": complete_text,
            }
            self.metrics.turns.append(turn_data)

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] AGENT: {complete_text}\n")

            self.metrics.agent_current_transcript = ""

        # Finalize any pending audio segment (even if no transcript received yet)
        # This ensures all recorded audio has a corresponding segment entry
        if self.audio_file and self.metrics.current_segment_start is not None:
            segment_start = self.metrics.current_segment_start - (self.metrics.audio_start_timestamp or 0)
            segment_end = timestamp - (self.metrics.audio_start_timestamp or 0)

            # Determine which speaker based on recent audio activity
            # If neither has transcript, use the most recent audio time to determine speaker
            if user_has_transcript:
                speaker = "user"
                transcript = self.metrics.user_current_transcript or "[incomplete]"
            elif agent_has_transcript:
                speaker = "agent"
                transcript = self.metrics.agent_current_transcript or "[incomplete]"
            else:
                # No transcript available, determine by last audio time
                user_last = self.metrics.user_last_audio_time or 0
                agent_last = self.metrics.agent_last_audio_time or 0
                if agent_last > user_last:
                    speaker = "agent"
                    transcript = "[no transcript received]"
                    logger.info(f"[AGENT FINAL] Finalizing audio segment with no transcript")
                else:
                    speaker = "user"
                    transcript = "[no transcript received]"
                    logger.info(f"[USER FINAL] Finalizing audio segment with no transcript")

            segment = SegmentEntry(
                start_time=segment_start, end_time=segment_end, speaker=speaker, transcript=transcript
            )
            self.metrics.segments.append(segment)
            logger.info(f"[SEGMENT FINAL] Created final segment: {speaker} {segment_start:.3f}-{segment_end:.3f}s")
            self.metrics.current_segment_start = None

        # Debug: Save accumulated sent audio chunks for analysis
        self._save_sent_audio_debug()

    def _save_sent_audio_debug(self):
        """Save final sent audio chunks to disk for debugging."""
        import os
        import wave

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "./eval_results/debug_sent_audio"
        os.makedirs(output_dir, exist_ok=True)

        # Save USER→AGENT final sent audio
        if self.sent_to_agent_chunks:
            filename = os.path.join(output_dir, f"sent_to_agent_{timestamp}.wav")
            audio_data = b"".join(self.sent_to_agent_chunks)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.agent_input_sample_rate)
                wav_file.writeframes(audio_array.tobytes())

            duration = len(audio_array) / self.agent_input_sample_rate
            logger.info(f"[DEBUG] Saved USER→AGENT sent audio: {filename}")
            logger.info(
                f"        Chunks: {len(self.sent_to_agent_chunks)}, Duration: {duration:.2f}s, Sample rate: {self.agent_input_sample_rate}Hz"
            )

        # Save AGENT→USER final sent audio
        if self.sent_to_user_chunks:
            filename = os.path.join(output_dir, f"sent_to_user_{timestamp}.wav")
            audio_data = b"".join(self.sent_to_user_chunks)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.user_input_sample_rate)
                wav_file.writeframes(audio_array.tobytes())

            duration = len(audio_array) / self.user_input_sample_rate
            logger.info(f"[DEBUG] Saved AGENT→USER sent audio: {filename}")
            logger.info(
                f"        Chunks: {len(self.sent_to_user_chunks)}, Duration: {duration:.2f}s, Sample rate: {self.user_input_sample_rate}Hz"
            )

    async def _monitor_user_message(self, frame):
        """
        Monitor user messages for timing and transcripts.
        This tracks when user sends audio and stops speaking.

        Args:
            frame: Already deserialized frame from the receive loop
        """
        timestamp = asyncio.get_event_loop().time()

        # Check if frame is valid
        if frame is None:
            return

        # Handle audio frames
        if hasattr(frame, 'audio') and frame.audio:
            self.metrics.user_last_audio_time = timestamp

            # Record audio if audio_file is specified
            if self.audio_file:
                # Initialize audio start timestamp on first audio
                if self.metrics.audio_start_timestamp is None:
                    self.metrics.audio_start_timestamp = timestamp
                    self.metrics.current_segment_start = timestamp
                    logger.debug(f"[USER AUDIO] Started recording user audio at {timestamp:.3f}")

                # Save raw audio data
                self.metrics.user_audio_chunks.append(frame.audio)
                logger.debug(
                    f"[USER AUDIO] Received user audio chunk: {len(frame.audio)} bytes (total chunks: {len(self.metrics.user_audio_chunks)})"
                )

            return

        # Handle message frames (RTVI messages)
        if hasattr(frame, 'message') and frame.message:
            # frame.message is already a dict, not a JSON string
            if isinstance(frame.message, str):
                data = json.loads(frame.message)
            else:
                data = frame.message
            message_type = data.get("type", "")

            # Track user transcription segments (accumulate)
            if message_type == RTVI_BOT_TRANSCRIPTION:
                text = data.get("data", {}).get("text", "")
                if text:
                    # Accumulate text segments (they arrive incrementally)
                    self.metrics.user_current_transcript += text
                    logger.debug(f"[USER SEGMENT] {text}")

            # Track when user bot stops speaking (finalize turn)
            elif message_type == RTVI_BOT_STOPPED_SPEAKING:
                self.metrics.user_last_audio_time = timestamp
                self.metrics.waiting_for_agent_response = True
                logger.debug(f"[TIMING] User stopped speaking at {timestamp:.3f}")

                # Finalize the turn with accumulated transcript
                if self.metrics.user_current_transcript:
                    complete_text = self.metrics.user_current_transcript.strip()
                    self.metrics.last_user_transcript = complete_text
                    logger.info(f"[USER] {complete_text}")

                    turn_data = {
                        "timestamp": datetime.now().isoformat(),
                        "role": "user",
                        "text": complete_text,
                    }
                    self.metrics.turns.append(turn_data)

                    if self.log_file:
                        with open(self.log_file, "a") as f:
                            f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] USER: {complete_text}\n")

                    # Create segment entry for segLST
                    if self.audio_file and self.metrics.current_segment_start is not None:
                        segment_start = self.metrics.current_segment_start - (self.metrics.audio_start_timestamp or 0)
                        segment_end = timestamp - (self.metrics.audio_start_timestamp or 0)
                        segment = SegmentEntry(
                            start_time=segment_start,
                            end_time=segment_end,
                            speaker="user",
                            transcript=complete_text,
                        )
                        self.metrics.segments.append(segment)
                        self.metrics.current_segment_start = None

                    # Clear accumulated text for next turn
                    self.metrics.user_current_transcript = ""

    async def _monitor_agent_message(self, frame):
        """
        Monitor agent messages for timing and transcripts.
        This tracks when agent starts responding (first audio received).

        Args:
            frame: Already deserialized frame from the receive loop
        """
        timestamp = asyncio.get_event_loop().time()

        # Check if frame is valid
        if frame is None:
            return

        # Debug: log frame type
        logger.debug(f"[AGENT MONITOR] Frame type: {type(frame).__name__}, has audio: {hasattr(frame, 'audio')}")

        # Handle audio frames - this is the response!
        if hasattr(frame, 'audio') and frame.audio:
            # If we're waiting for agent response and this is the first audio
            if self.metrics.waiting_for_agent_response and self.metrics.user_last_audio_time:
                latency_ms = (timestamp - self.metrics.user_last_audio_time) * 1000

                # Create latency measurement
                latency = ResponseLatency(
                    user_stop_time=self.metrics.user_last_audio_time,
                    agent_start_time=timestamp,
                    latency_ms=latency_ms,
                    user_transcript=self.metrics.last_user_transcript,
                )

                self.metrics.latencies.append(latency)
                self.metrics.waiting_for_agent_response = False

                logger.info(f"[LATENCY] Response latency: {latency_ms:.1f}ms")

                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"  → Response latency: {latency_ms:.1f}ms\n")

                # Track segment start for agent when it starts responding
                if self.audio_file and self.metrics.current_segment_start is None:
                    self.metrics.current_segment_start = timestamp

            self.metrics.agent_last_audio_time = timestamp

            # Record audio if audio_file is specified
            if self.audio_file:
                # Initialize audio start timestamp on first audio
                if self.metrics.audio_start_timestamp is None:
                    self.metrics.audio_start_timestamp = timestamp
                    self.metrics.current_segment_start = timestamp
                    logger.debug(f"[AGENT AUDIO] Started recording agent audio at {timestamp:.3f}")

                # Save raw audio data
                self.metrics.agent_audio_chunks.append(frame.audio)
                audio_len_in_seconds = len(frame.audio) / 2 / self.agent_output_sample_rate
                logger.debug(
                    f"[AGENT AUDIO] Received agent audio chunk: {len(frame.audio)} bytes of {audio_len_in_seconds:.2f} seconds"
                )

            return

        # Handle message frames (RTVI messages)
        message_type = ""
        if hasattr(frame, 'message') and frame.message:
            # frame.message is already a dict, not a JSON string
            if isinstance(frame.message, str):
                data = json.loads(frame.message)
            else:
                data = frame.message
            message_type = data.get("type", "")

        # Track when agent bot starts speaking
        if message_type == RTVI_BOT_STARTED_SPEAKING:
            if self.metrics.waiting_for_agent_response and self.metrics.user_last_audio_time:
                latency_ms = (timestamp - self.metrics.user_last_audio_time) * 1000

                logger.debug(f"[TIMING] Agent started speaking at {timestamp:.3f} (latency: {latency_ms:.1f}ms)")

        # Track agent transcription segments (accumulate)
        elif message_type == RTVI_BOT_TRANSCRIPTION:
            text = data.get("data", {}).get("text", "")
            if text:
                # Accumulate text segments (they arrive incrementally)
                self.metrics.agent_current_transcript += text
                logger.debug(f"[AGENT SEGMENT] {text}")

        # Track when agent bot stops speaking (finalize turn)
        elif message_type == RTVI_BOT_STOPPED_SPEAKING:
            logger.debug(f"[TIMING] Agent stopped speaking at {timestamp:.3f}")

            # Finalize the turn with accumulated transcript
            if self.metrics.agent_current_transcript:
                complete_text = self.metrics.agent_current_transcript.strip()
                logger.info(f"[AGENT] {complete_text}")

                # Update the last latency measurement with complete agent transcript
                if self.metrics.latencies and not self.metrics.latencies[-1].agent_transcript:
                    self.metrics.latencies[-1].agent_transcript = complete_text

                turn_data = {
                    "timestamp": datetime.now().isoformat(),
                    "role": "agent",
                    "text": complete_text,
                }
                self.metrics.turns.append(turn_data)

                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] AGENT: {complete_text}\n")

                # Create segment entry for segLST
                if self.audio_file and self.metrics.current_segment_start is not None:
                    segment_start = self.metrics.current_segment_start - (self.metrics.audio_start_timestamp or 0)
                    segment_end = timestamp - (self.metrics.audio_start_timestamp or 0)
                    segment = SegmentEntry(
                        start_time=segment_start, end_time=segment_end, speaker="agent", transcript=complete_text
                    )
                    self.metrics.segments.append(segment)
                    self.metrics.current_segment_start = None

                # Clear accumulated text for next turn
                self.metrics.agent_current_transcript = ""

    def _resample_audio_for_saving(self, audio_chunks: List[bytes], from_rate: int, to_rate: int) -> np.ndarray:
        """
        Resample audio chunks from one sample rate to another.

        Args:
            audio_chunks: List of audio byte chunks (16-bit PCM)
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled audio as numpy array
        """
        if not audio_chunks:
            return np.array([], dtype=np.int16)

        # Concatenate all chunks
        audio_bytes = b''.join(audio_chunks)

        # Convert bytes to int16 array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # If sample rates match, no resampling needed
        if from_rate == to_rate:
            return audio_array

        # Simple linear resampling
        duration = len(audio_array) / from_rate
        target_length = int(duration * to_rate)

        # Use numpy interp for resampling
        x_old = np.linspace(0, duration, len(audio_array))
        x_new = np.linspace(0, duration, target_length)
        resampled = np.interp(x_new, x_old, audio_array)

        return resampled.astype(np.int16)

    def save_audio_and_seglst(self, audio_file: str = None):
        """Save stereo audio file and segLST transcript file."""
        if audio_file:
            self.audio_file = audio_file
        if not self.audio_file:
            return
        try:
            logger.info(f"Saving audio to {self.audio_file}...")

            # Resample both channels to output sample rate
            user_audio = self._resample_audio_for_saving(
                self.metrics.user_audio_chunks, self.user_output_sample_rate, self.output_sample_rate
            )
            agent_audio = self._resample_audio_for_saving(
                self.metrics.agent_audio_chunks, self.agent_output_sample_rate, self.output_sample_rate
            )

            # Make both arrays the same length (pad shorter one with zeros)
            max_length = max(len(user_audio), len(agent_audio))
            if len(user_audio) < max_length:
                user_audio = np.pad(user_audio, (0, max_length - len(user_audio)))
            if len(agent_audio) < max_length:
                agent_audio = np.pad(agent_audio, (0, max_length - len(agent_audio)))

            # Create stereo array (user=left, agent=right)
            stereo_audio = np.empty((max_length, 2), dtype=np.int16)
            stereo_audio[:, 0] = user_audio  # Left channel
            stereo_audio[:, 1] = agent_audio  # Right channel

            # Save as WAV file
            with wave.open(self.audio_file, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.output_sample_rate)
                wav_file.writeframes(stereo_audio.tobytes())

            logger.info(f"Audio saved: {self.audio_file}")
            logger.info(f"  Channels: 2 (user=left, agent=right)")
            logger.info(f"  Sample rate: {self.output_sample_rate} Hz")
            logger.info(f"  Duration: {max_length / self.output_sample_rate:.2f}s")

            # Save segLST file
            seglst_file = Path(self.audio_file).with_suffix('.seglst')
            with open(seglst_file, 'w') as f:
                f.write("# segLST format: start_time end_time speaker transcript\n")
                f.write(f"# Audio file: {Path(self.audio_file).name}\n")
                f.write(f"# Sample rate: {self.output_sample_rate} Hz\n")
                f.write("#\n")

                # Write segments sorted by start time
                sorted_segments = sorted(self.metrics.segments, key=lambda s: s.start_time)
                for seg in sorted_segments:
                    # Format: start end speaker text
                    f.write(f"{seg.start_time:.3f} {seg.end_time:.3f} {seg.speaker} {seg.transcript}\n")

            logger.info(f"segLST saved: {seglst_file}")
            logger.info(f"  Total segments: {len(self.metrics.segments)}")

        except Exception as e:
            logger.error(f"Error saving audio/segLST: {e}")
            import traceback

            traceback.print_exc()

    async def disconnect(self):
        """Disconnect from both user and agent"""
        self.metrics.end_time = datetime.now()

        if self.user_ws:
            await self.user_ws.close()
        if self.agent_ws:
            await self.agent_ws.close()

        logger.info("Disconnected from user and agent")

        # Log final statistics
        latency_stats = self.metrics.get_latency_stats()
        if latency_stats['count'] > 0:
            logger.info(f"\nFinal Latency Statistics:")
            logger.info(f"  Measurements: {latency_stats['count']}")
            logger.info(f"  Mean: {latency_stats['mean_ms']:.1f}ms")
            logger.info(f"  Median: {latency_stats['median_ms']:.1f}ms")
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
