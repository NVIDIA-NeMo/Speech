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
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import websockets
from loguru import logger
from omegaconf import DictConfig
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.processors.frameworks.rtvi import (
    RTVIBotStartedSpeakingMessage,
    RTVIBotStoppedSpeakingMessage,
    RTVIBotTranscriptionMessage,
    RTVIBotTTSTextMessage,
    RTVITextMessageData,
)
from pipecat.serializers.protobuf import MessageFrame, ProtobufFrameSerializer

from nemo.agents.voice_agent.utils import setup_logging

# Import AudioStream for buffering and resampling
from nemo.agents.voice_agent.utils.audio import AudioStream, NoiseConfig

# RTVI message type constants - automatically adapts to pipecat changes
RTVI_BOT_STOPPED_SPEAKING = RTVIBotStoppedSpeakingMessage().type
RTVI_BOT_STARTED_SPEAKING = RTVIBotStartedSpeakingMessage().type
RTVI_BOT_TRANSCRIPTION = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text="")).type
RTVI_BOT_TTS_TEXT = RTVIBotTTSTextMessage(data=RTVITextMessageData(text="")).type


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
    audio_chunks: List[bytes] = field(default_factory=list)  # Audio data for this segment


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

    # Track if agent bot is currently speaking
    agent_bot_is_speaking: bool = False

    # Audio recording (for stereo WAV output)
    user_audio_chunks: List[bytes] = field(default_factory=list)
    agent_audio_chunks: List[bytes] = field(default_factory=list)
    audio_sample_rate: int = 16000  # Default sample rate
    audio_start_timestamp: Optional[float] = None  # When first audio arrives (for audio file creation)
    thread_start_timestamp: Optional[float] = None  # When routing threads start (for conversation log timing)

    # Segment tracking for segLST output
    segments: List[SegmentEntry] = field(default_factory=list)
    current_user_segment: Optional[SegmentEntry] = None
    current_agent_segment: Optional[SegmentEntry] = None

    # Audio-based segment tracking (more accurate timing)
    audio_segments: List[SegmentEntry] = field(default_factory=list)
    audio_user_segment_in_progress: Optional[SegmentEntry] = None
    audio_agent_segment_in_progress: Optional[SegmentEntry] = None
    audio_user_last_speech_time: Optional[float] = None
    audio_agent_last_speech_time: Optional[float] = None

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
        self.agent_bot_is_speaking = False

        # Reset scenario-specific metrics (for multi-scenario evaluations)
        self.latencies = []
        self.turns = []
        self.segments = []
        self.log_entries = []

        # Reset audio buffers
        self.user_audio_chunks = []
        self.agent_audio_chunks = []
        self.audio_start_timestamp = None
        self.thread_start_timestamp = None
        self.current_user_segment = None
        self.current_agent_segment = None

        # Reset audio-based segment tracking
        self.audio_segments = []
        self.audio_user_segment_in_progress = None
        self.audio_agent_segment_in_progress = None
        self.audio_user_last_speech_time = None
        self.audio_agent_last_speech_time = None


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
        grace_period: float = 5.0,
        seglst_start_offset_seconds: float = -0.00,
        seglst_end_offset_seconds: float = -0.00,
        turn_end_silence_threshold: float = 0.35,
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
            burst_size_range: Range of the burst size
            burst_delay_ms: Delay between the frames in the burst
            grace_period: Grace period after the main duration, used to drain the websocket
            seglst_start_offset_seconds: Start offset for the segLST, used to adjust the start time of the segLST timestamps
            seglst_end_offset_seconds: End offset for the segLST, used to adjust the end time of the segLST timestamps
            turn_end_silence_threshold: Silence threshold for the turn, used to determine the end-of-turn by continuous silence
            noise_config: Noise configuration, used to configure the noise for the audio stream
        """
        self.user_url = user_url
        self.agent_url = agent_url
        self.output_dir = output_dir
        self.scenario_name = scenario_name
        self.log_file = None
        self.audio_file = None
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

        # Audio-based turn detection configuration
        self.turn_end_silence_threshold = turn_end_silence_threshold  # Continuous silence before ending turn

        self.seglst_start_offset_seconds = (
            seglst_start_offset_seconds  # Default additional offset for segLST timestamps
        )
        self.seglst_end_offset_seconds = seglst_end_offset_seconds  # Default additional offset for segLST timestamps

        # Noise configuration for user channel
        self.noise_config = noise_config

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
        else:
            logger.info(f"Steady mode: sending at constant {self.audio_chunk_in_seconds * 1000:.0f}ms intervals")

        # Initialize output directory and log files
        if output_dir:
            self.init_output_dir(output_dir, scenario_name, log_level)

        self.bridge_ready = False

    def init_output_dir(self, output_dir: str, scenario_name: Optional[str] = None, log_level: str = "DEBUG"):
        """Initialize the output directory and all derived log/audio file paths."""
        logger.info(f"Initializing output directory: {output_dir}, session name: {scenario_name}")
        self.output_dir = output_dir
        self.scenario_name = scenario_name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = str(Path(output_dir) / "conversation_log.txt")
        self.audio_file = str(Path(output_dir) / "concat_audio_segments.wav")
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
        await self.update_user_prompt(scenario["user_prompt"], auto_reset=False)

        if "agent_prompt" in scenario:
            # Note: update_system_prompt handler in bot_websocket.py automatically resets,
            # so we don't need explicit reset calls here
            await self.update_agent_prompt(scenario["agent_prompt"], auto_reset=False)
        else:
            # reset agent cache
            await self.reset_agent()

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

    def _track_audio_segment(self, direction: str, has_speech: bool, audio_chunk: bytes, timestamp: float):
        """
        Track audio segments based on has_speech signal from AudioStream.
        Creates segment on speech start, closes on sustained silence.

        Args:
            direction: "USER→AGENT" or "AGENT→USER"
            has_speech: Whether the audio chunk contains speech (from AudioStream)
            audio_chunk: The audio bytes
            timestamp: Absolute timestamp (asyncio loop time)
        """
        if direction == "USER→AGENT":
            segment = self.metrics.audio_user_segment_in_progress
            last_speech = self.metrics.audio_user_last_speech_time
            speaker = "user"
        else:  # AGENT→USER
            segment = self.metrics.audio_agent_segment_in_progress
            last_speech = self.metrics.audio_agent_last_speech_time
            speaker = "agent"

        if has_speech:
            # Update last speech time
            if direction == "USER→AGENT":
                self.metrics.audio_user_last_speech_time = timestamp
            else:
                self.metrics.audio_agent_last_speech_time = timestamp

            # Start new segment if not already in progress
            if segment is None:
                relative_time = self._get_relative_time(timestamp)
                logger.info(f"[AUDIO SEG] {speaker.capitalize()} segment started at {relative_time:.3f}s")

                segment = SegmentEntry(
                    start_time=relative_time,
                    end_time=relative_time,  # Will be updated
                    speaker=speaker,
                    transcript="",  # Will be filled from protocol
                    audio_chunks=[],
                )

                if direction == "USER→AGENT":
                    self.metrics.audio_user_segment_in_progress = segment
                else:
                    self.metrics.audio_agent_segment_in_progress = segment

            # Append audio to current segment
            if segment is not None:
                segment.audio_chunks.append(audio_chunk)
                segment.end_time = self._get_relative_time(timestamp)

        else:  # No speech (silence/noise)
            # Check if we should close the segment
            if segment is not None and last_speech is not None:
                silence_duration = timestamp - last_speech

                if silence_duration >= self.turn_end_silence_threshold:
                    # Close segment after sustained silence
                    segment.end_time = self._get_relative_time(last_speech)

                    # Ensure end_time is not before start_time (can happen with timing jitter)
                    if segment.end_time < segment.start_time:
                        logger.warning(
                            f"[AUDIO SEG] {speaker.capitalize()} segment has end_time < start_time, "
                            f"adjusting: {segment.start_time:.3f}s -> {segment.end_time:.3f}s"
                        )
                        segment.end_time = segment.start_time

                    duration = segment.end_time - segment.start_time

                    logger.info(
                        f"[AUDIO SEG] {speaker.capitalize()} segment ended at {segment.end_time:.3f}s "
                        f"(duration: {duration:.3f}s, {len(segment.audio_chunks)} chunks)"
                    )

                    # Only save segments with audio and valid duration
                    if len(segment.audio_chunks) > 0 and duration >= 0:
                        self.metrics.audio_segments.append(segment)

                    # Clear in-progress segment
                    if direction == "USER→AGENT":
                        self.metrics.audio_user_segment_in_progress = None
                        self.metrics.audio_user_last_speech_time = None
                    else:
                        self.metrics.audio_agent_segment_in_progress = None
                        self.metrics.audio_agent_last_speech_time = None

    def _match_transcript_to_audio_segment(self, speaker: str, transcript: str, protocol_timestamp: float):
        """
        Match a transcript from protocol message to the nearest audio segment.

        Args:
            speaker: "user" or "agent"
            transcript: The transcript text from protocol message
            protocol_timestamp: Timestamp when protocol message was received
        """
        # Find audio segment that overlaps with protocol timestamp
        matching_segment = None
        min_distance = float('inf')

        for seg in self.metrics.audio_segments:
            if seg.speaker == speaker and seg.transcript == "":  # Unfilled
                # Check if protocol timestamp falls within segment
                if seg.start_time <= protocol_timestamp <= seg.end_time:
                    matching_segment = seg
                    break

                # Or find nearest segment (in case of timing mismatch)
                distance = min(abs(seg.start_time - protocol_timestamp), abs(seg.end_time - protocol_timestamp))
                if distance < min_distance:
                    min_distance = distance
                    matching_segment = seg

        # Also check in-progress segments
        if matching_segment is None:
            if speaker == "user" and self.metrics.audio_user_segment_in_progress is not None:
                seg = self.metrics.audio_user_segment_in_progress
                if seg.transcript == "" and seg.start_time <= protocol_timestamp:
                    matching_segment = seg
            elif speaker == "agent" and self.metrics.audio_agent_segment_in_progress is not None:
                seg = self.metrics.audio_agent_segment_in_progress
                if seg.transcript == "" and seg.start_time <= protocol_timestamp:
                    matching_segment = seg

        if matching_segment is not None:
            matching_segment.transcript = transcript
            logger.info(
                f"[TRANSCRIPT MATCH] Matched '{transcript[:30]}...' to "
                f"{speaker} segment at {matching_segment.start_time:.3f}s"
            )
        else:
            logger.warning(
                f"[TRANSCRIPT MATCH] No audio segment found for {speaker} "
                f"transcript at {protocol_timestamp:.3f}s: '{transcript[:30]}...'"
            )

    def _finalize_audio_segments(self):
        """Close any in-progress audio segments at end of evaluation."""
        if self.metrics.thread_start_timestamp is None:
            return

        loop = asyncio.get_event_loop()
        current_time = loop.time()

        # Finalize user segment
        if self.metrics.audio_user_segment_in_progress is not None:
            seg = self.metrics.audio_user_segment_in_progress
            seg.end_time = self._get_relative_time(current_time)
            if len(seg.audio_chunks) > 0:
                logger.info(
                    f"[AUDIO SEG] Finalizing user segment at {seg.end_time:.3f}s " f"({len(seg.audio_chunks)} chunks)"
                )
                self.metrics.audio_segments.append(seg)
            self.metrics.audio_user_segment_in_progress = None

        # Finalize agent segment
        if self.metrics.audio_agent_segment_in_progress is not None:
            seg = self.metrics.audio_agent_segment_in_progress
            seg.end_time = self._get_relative_time(current_time)
            if len(seg.audio_chunks) > 0:
                logger.info(
                    f"[AUDIO SEG] Finalizing agent segment at {seg.end_time:.3f}s " f"({len(seg.audio_chunks)} chunks)"
                )
                self.metrics.audio_segments.append(seg)
            self.metrics.audio_agent_segment_in_progress = None

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
        if self.user_ws:
            await self._send_reset_action(self.user_ws, "user")
        if self.agent_ws:
            await self._send_reset_action(self.agent_ws, "agent")

        # Reset all metrics
        self.metrics.reset()

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
            async for message in ws:
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
                    logger.debug(f"[{direction}] In grace period, skip monitoring message: {frame}")
                    continue

                # Monitor messages
                await monitor_func(frame)

                # Check if this is audio
                if hasattr(frame, 'audio') and frame.audio:
                    # Put raw audio into thread-safe queue
                    queue.put(frame.audio)
                    logger.debug(f"[{direction}] Queued {len(frame.audio)} bytes of audio")

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
        chunk_count = 0
        target_time = start_time  # Track target time incrementally for numerical stability

        try:
            while True:
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

                logger.debug(f"[{direction}] Retrieved {chunks_retrieved} chunks from queue")
                if in_grace_period:
                    logger.debug(f"[{direction}] In grace period, skip forwarding audio: {chunks_retrieved} chunks")
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

                    # Track audio-based segments
                    self._track_audio_segment(direction, has_speech, audio_to_send, loop.time())

                    # Track sent audio
                    sent_chunks_list.append(audio_to_send)

                    # Create frame and send
                    output_frame = OutputAudioRawFrame(
                        audio=audio_to_send, sample_rate=audio_stream.output_sample_rate, num_channels=1
                    )
                    serialized = await self.serializer.serialize(output_frame)
                    await dest_ws.send(serialized)

                    chunk_count += 1

                    logger.debug(
                        f"[{direction}][{chunk_count}] Sent {len(audio_to_send)} bytes ({idx+1}/{burst_size}, has_speech: {has_speech})"
                    )

                # Time-based scheduling: increment target time from previous burst
                # This automatically compensates for processing overhead and is numerically stable
                target_time += burst_size * self.audio_chunk_in_seconds
                current_time = loop.time()
                wait_duration = max(0.001, target_time - current_time)

                if wait_duration < 0.001:
                    logger.debug(f"[{direction}] Behind schedule by {-wait_duration:.3f}s after {chunk_count} frames")

                if self.use_burst_mode:
                    logger.debug(
                        f"[{direction}] Burst complete ({burst_size} frames), waiting {wait_duration*1000:.1f}ms (target: {target_time:.3f}s)"
                    )
                await asyncio.sleep(wait_duration)

            logger.info(f"[{direction}] Send loop finished after {chunk_count} chunks")

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

            except Exception as e:
                logger.error(f"[AGENT THREAD] Error: {e}", exc_info=True)
            finally:
                logger.info("[AGENT THREAD] Exiting")

        try:
            loop.run_until_complete(agent_loop())
        finally:
            loop.close()

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
            raise RuntimeError("[EVAL BRIDGE] Bridge is not ready, please call `bridge.prepare_for_scenario()` first")

        logger.info(f"[EVAL BRIDGE] Running scenario for {duration} seconds...")
        self.metrics.start_time = datetime.now()

        # Clear debug accumulation lists for this run (only final sent audio)
        self.sent_to_agent_chunks = []
        self.sent_to_user_chunks = []

        # Clear thread-safe queues
        self.user_to_agent_queue = queue.Queue()
        self.agent_to_user_queue = queue.Queue()

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

        # Set thread start timestamp for conversation log timing (aligns with bridge_audio_log.wav)
        loop = asyncio.get_event_loop()
        self.metrics.thread_start_timestamp = loop.time()

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
        loop = asyncio.get_event_loop()
        timestamp = loop.time()  # Use asyncio time to match other timestamps

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

            # Calculate turn timing from current_user_segment
            if self.metrics.current_user_segment is not None:
                segment_start = self.metrics.current_user_segment.start_time
                segment_end = self._get_relative_time(timestamp)
            else:
                segment_start = 0.0
                segment_end = self._get_relative_time(timestamp)

            if self.log_file:
                log_entry = self._format_turn_log("user", complete_text, segment_start, segment_end)
                self.metrics.log_entries.append((segment_start, log_entry))

            # Finalize user segment if one exists
            if self.audio_file and self.metrics.current_user_segment is not None:
                self.metrics.current_user_segment.end_time = segment_end
                self.metrics.current_user_segment.transcript = complete_text
                self.metrics.segments.append(self.metrics.current_user_segment)
                self.metrics.current_user_segment = None

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

            # Calculate turn timing from current_agent_segment
            if self.metrics.current_agent_segment is not None:
                segment_start = self.metrics.current_agent_segment.start_time
                segment_end = self._get_relative_time(timestamp)
            else:
                segment_start = 0.0
                segment_end = self._get_relative_time(timestamp)

            # Get latency if available
            latency_ms = None
            if self.metrics.latencies and self.metrics.latencies[-1].agent_transcript == complete_text:
                latency_ms = self.metrics.latencies[-1].latency_ms

            if self.log_file:
                log_entry = self._format_turn_log("agent", complete_text, segment_start, segment_end, latency_ms)
                self.metrics.log_entries.append((segment_start, log_entry))

            # Finalize agent segment if one exists
            if self.audio_file and self.metrics.current_agent_segment is not None:
                self.metrics.current_agent_segment.end_time = segment_end
                self.metrics.current_agent_segment.transcript = complete_text
                self.metrics.segments.append(self.metrics.current_agent_segment)
                self.metrics.current_agent_segment = None

            self.metrics.agent_current_transcript = ""

        # Finalize any in-progress audio segments
        self._finalize_audio_segments()

        # Write sorted log entries to conversation log
        self._write_sorted_log_entries()

        # Debug: Save accumulated sent audio chunks for analysis
        self._save_bridge_audio_log()
        self._save_audio_and_seglst()

    def _write_sorted_log_entries(self):
        """Write all buffered log entries to file, sorted by start time."""
        if not self.log_file or not self.metrics.log_entries:
            return

        try:
            # Sort entries by start_time (first element of tuple)
            sorted_entries = sorted(self.metrics.log_entries, key=lambda x: x[0])

            # Append all sorted entries to log file
            with open(self.log_file, "a") as f:
                for _start_time, log_entry in sorted_entries:
                    f.write(log_entry)

            logger.info(f"[LOG] Wrote {len(sorted_entries)} conversation turns to log file (sorted by time)")
        except Exception as e:
            logger.error(f"[LOG] Error writing sorted log entries: {e}")

    def _save_bridge_audio_log(self):
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

        if len(channel0) > 0 and self.agent_input_sample_rate != target_rate:
            # Resample channel 0
            resample_ratio = target_rate / self.agent_input_sample_rate
            new_length = int(len(channel0) * resample_ratio)
            channel0 = np.interp(
                np.linspace(0, len(channel0) - 1, new_length), np.arange(len(channel0)), channel0
            ).astype(np.int16)

        if len(channel1) > 0 and self.user_input_sample_rate != target_rate:
            # Resample channel 1
            resample_ratio = target_rate / self.user_input_sample_rate
            new_length = int(len(channel1) * resample_ratio)
            channel1 = np.interp(
                np.linspace(0, len(channel1) - 1, new_length), np.arange(len(channel1)), channel1
            ).astype(np.int16)

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

        logger.debug(
            f"[USER MONITOR] Frame type: {type(frame).__name__}, has audio: {hasattr(frame, 'audio')}, frame: {frame}"
        )

        # Handle audio frames
        if hasattr(frame, 'audio') and frame.audio:
            self.metrics.user_last_audio_time = timestamp

            # Record audio if audio_file is specified
            if self.audio_file:
                # Initialize audio start timestamp on first audio (from any speaker)
                if self.metrics.audio_start_timestamp is None:
                    self.metrics.audio_start_timestamp = timestamp
                    logger.debug(f"[USER AUDIO] Started recording audio at {timestamp:.3f}")

                # Save raw audio data
                self.metrics.user_audio_chunks.append(frame.audio)

                # Also append to current segment if one is active
                if self.metrics.current_user_segment is not None:
                    self.metrics.current_user_segment.audio_chunks.append(frame.audio)

                logger.debug(
                    f"[USER AUDIO] Received user audio chunk: {len(frame.audio)} bytes of {len(frame.audio) / 2 / self.user_output_sample_rate:.4f} seconds"
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

            # Track when user bot starts speaking
            if message_type == RTVI_BOT_STARTED_SPEAKING:
                # Create a new segment for this turn
                if self.audio_file:
                    relative_time = self._get_relative_time(timestamp)
                    self.metrics.current_user_segment = SegmentEntry(
                        start_time=relative_time,
                        end_time=relative_time,  # Will be updated on stop
                        speaker="user",
                        transcript="",  # Will be filled when text arrives
                    )
                logger.debug("[TIMING] User started speaking")
            # Track user TTS text segments (accumulate)
            elif message_type == RTVI_BOT_TTS_TEXT:
                text = data.get("data", {}).get("text", "")
                if text:
                    # Accumulate text segments (they arrive incrementally)
                    self.metrics.user_current_transcript += text
                    logger.debug(f"[USER SEGMENT]: {text}")
            # Track when user bot stops speaking (finalize turn)
            elif message_type == RTVI_BOT_STOPPED_SPEAKING:
                self.metrics.user_last_audio_time = timestamp
                self.metrics.waiting_for_agent_response = True
                logger.debug(f"[TIMING] User stopped speaking at {timestamp:.3f}")

                # Finalize the turn with accumulated transcript
                if self.metrics.user_current_transcript:
                    complete_text = self.metrics.user_current_transcript.strip()
                    self.metrics.last_user_transcript = complete_text
                    logger.info(f"[USER] Finalized Text: {complete_text}")

                    # Match transcript to audio segment
                    protocol_time = self._get_relative_time(timestamp)
                    self._match_transcript_to_audio_segment("user", complete_text, protocol_time)

                    turn_data = {
                        "timestamp": datetime.now().isoformat(),
                        "role": "user",
                        "text": complete_text,
                    }
                    self.metrics.turns.append(turn_data)

                    # Find the matched audio segment for accurate timing
                    matched_audio_segment = None
                    for seg in self.metrics.audio_segments:
                        if seg.speaker == "user" and seg.transcript == complete_text:
                            matched_audio_segment = seg
                            break

                    # If no match in completed segments, check in-progress segment
                    if not matched_audio_segment and self.metrics.audio_user_segment_in_progress:
                        if self.metrics.audio_user_segment_in_progress.transcript == complete_text:
                            matched_audio_segment = self.metrics.audio_user_segment_in_progress

                    # Use audio segment timing if available, otherwise fallback to protocol timing
                    if matched_audio_segment:
                        segment_start = matched_audio_segment.start_time
                        segment_end = matched_audio_segment.end_time
                    else:
                        # Fallback to protocol-based timing from current_user_segment
                        if self.metrics.current_user_segment is not None:
                            segment_start = self.metrics.current_user_segment.start_time
                            segment_end = self._get_relative_time(timestamp)
                        else:
                            # Fallback if segment wasn't created
                            segment_start = 0.0
                            segment_end = self._get_relative_time(timestamp)

                    if self.log_file:
                        log_entry = self._format_turn_log("user", complete_text, segment_start, segment_end)
                        self.metrics.log_entries.append((segment_start, log_entry))

                    # Create segment entry for segLST
                    if self.audio_file and self.metrics.current_user_segment is not None:
                        # Finalize the current segment with transcript and end time
                        self.metrics.current_user_segment.end_time = segment_end
                        self.metrics.current_user_segment.transcript = complete_text
                        self.metrics.segments.append(self.metrics.current_user_segment)
                        self.metrics.current_user_segment = None
                    elif self.audio_file and segment_start != 0.0:
                        # Fallback: create segment without audio chunks (shouldn't happen normally)
                        segment = SegmentEntry(
                            start_time=segment_start,
                            end_time=segment_end,
                            speaker="user",
                            transcript=complete_text,
                        )
                        self.metrics.segments.append(segment)

                    # Clear accumulated text for next turn
                    self.metrics.user_current_transcript = ""
                else:
                    # No transcript accumulated - response was likely interrupted
                    # Create placeholder entry since audio was still recorded
                    logger.warning(f"[USER] Stopped speaking but no transcript was accumulated (interrupted)")
                    complete_text = "[turn interrupted]"
                    self.metrics.last_user_transcript = complete_text
                    logger.info(f"[USER] Finalized Text: {complete_text}")

                    turn_data = {
                        "timestamp": datetime.now().isoformat(),
                        "role": "user",
                        "text": complete_text,
                    }
                    self.metrics.turns.append(turn_data)

                    # Calculate turn timing for log and segLST
                    if self.metrics.current_user_segment is not None:
                        segment_start = self.metrics.current_user_segment.start_time
                        segment_end = self._get_relative_time(timestamp)
                    else:
                        segment_start = 0.0
                        segment_end = self._get_relative_time(timestamp)

                    if self.log_file:
                        log_entry = self._format_turn_log("user", complete_text, segment_start, segment_end)
                        self.metrics.log_entries.append((segment_start, log_entry))

                    # Create segment entry for segLST with interrupted placeholder
                    if self.audio_file and self.metrics.current_user_segment is not None:
                        # Finalize the current segment with interrupted text
                        self.metrics.current_user_segment.end_time = segment_end
                        self.metrics.current_user_segment.transcript = complete_text
                        self.metrics.segments.append(self.metrics.current_user_segment)
                        self.metrics.current_user_segment = None
                    elif self.audio_file and segment_start != 0.0:
                        # Fallback: create segment without audio chunks
                        segment = SegmentEntry(
                            start_time=segment_start,
                            end_time=segment_end,
                            speaker="user",
                            transcript=complete_text,
                        )
                        self.metrics.segments.append(segment)

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
        logger.debug(
            f"[AGENT MONITOR] Frame type: {type(frame).__name__}, has audio: {hasattr(frame, 'audio')}, frame: {frame}"
        )

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

                # Note: Don't write latency to log file here - it will be included
                # in the agent's turn log when bot-stopped-speaking is received

            self.metrics.agent_last_audio_time = timestamp

            # Record audio if audio_file is specified
            if self.audio_file:
                # Initialize audio start timestamp on first audio (from any speaker)
                if self.metrics.audio_start_timestamp is None:
                    self.metrics.audio_start_timestamp = timestamp
                    logger.debug(f"[AGENT AUDIO] Started recording audio at {timestamp:.3f}")

                # Save raw audio data
                self.metrics.agent_audio_chunks.append(frame.audio)

                # Also append to current segment if one is active
                if self.metrics.current_agent_segment is not None:
                    self.metrics.current_agent_segment.audio_chunks.append(frame.audio)

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
            # If we have an accumulated transcript from a previous response, finalize it now
            if self.metrics.agent_current_transcript and not self.metrics.agent_bot_is_speaking:
                complete_text = self.metrics.agent_current_transcript.strip()
                logger.info(f"[AGENT] (auto-finalized on new start) {complete_text}")

                turn_data = {
                    "timestamp": datetime.now().isoformat(),
                    "role": "agent",
                    "text": complete_text,
                }
                self.metrics.turns.append(turn_data)

                # Find the matched audio segment for accurate timing
                matched_audio_segment = None
                for seg in self.metrics.audio_segments:
                    if seg.speaker == "agent" and seg.transcript == complete_text:
                        matched_audio_segment = seg
                        break

                # If no match in completed segments, check in-progress segment
                if not matched_audio_segment and self.metrics.audio_agent_segment_in_progress:
                    if self.metrics.audio_agent_segment_in_progress.transcript == complete_text:
                        matched_audio_segment = self.metrics.audio_agent_segment_in_progress

                # Use audio segment timing if available, otherwise fallback to protocol timing
                if matched_audio_segment:
                    segment_start = matched_audio_segment.start_time
                    segment_end = matched_audio_segment.end_time
                else:
                    # Fallback to protocol-based timing from current_agent_segment
                    if self.metrics.current_agent_segment is not None:
                        segment_start = self.metrics.current_agent_segment.start_time
                        segment_end = self._get_relative_time(timestamp)
                    else:
                        segment_start = 0.0
                        segment_end = self._get_relative_time(timestamp)

                # Calculate latency based on audio segment timing
                latency_ms = None
                if matched_audio_segment and self.metrics.audio_segments:
                    # Find the most recent user segment before this agent segment
                    for seg in reversed(self.metrics.audio_segments):
                        if seg.speaker == "user" and seg.end_time < matched_audio_segment.start_time:
                            latency_ms = (matched_audio_segment.start_time - seg.end_time) * 1000
                            break

                if self.log_file:
                    log_entry = self._format_turn_log("agent", complete_text, segment_start, segment_end, latency_ms)
                    self.metrics.log_entries.append((segment_start, log_entry))

                # Finalize segment if one exists
                if self.audio_file and self.metrics.current_agent_segment is not None:
                    self.metrics.current_agent_segment.end_time = segment_end
                    self.metrics.current_agent_segment.transcript = complete_text
                    self.metrics.segments.append(self.metrics.current_agent_segment)
                    self.metrics.current_agent_segment = None

                # Clear accumulated text
                self.metrics.agent_current_transcript = ""

            # Mark that agent is now speaking
            self.metrics.agent_bot_is_speaking = True

            # Create a new segment for this turn
            if self.audio_file:
                relative_time = self._get_relative_time(timestamp)
                self.metrics.current_agent_segment = SegmentEntry(
                    start_time=relative_time,
                    end_time=relative_time,  # Will be updated on stop
                    speaker="agent",
                    transcript="",  # Will be filled when text arrives
                )

            if self.metrics.waiting_for_agent_response and self.metrics.user_last_audio_time:
                latency_ms = (timestamp - self.metrics.user_last_audio_time) * 1000
                logger.debug(f"[TIMING] Agent started speaking at {timestamp:.3f} (latency: {latency_ms:.1f}ms)")
            else:
                logger.debug("[TIMING] Agent started speaking")

        # Track agent TTS text segments (accumulate)
        elif message_type == RTVI_BOT_TTS_TEXT:
            text = data.get("data", {}).get("text", "")
            if text:
                # Accumulate text segments (they arrive incrementally)
                self.metrics.agent_current_transcript += text
                logger.debug(f"[AGENT SEGMENT] {text}")

        # Track when agent bot stops speaking (finalize turn)
        elif message_type == RTVI_BOT_STOPPED_SPEAKING:
            logger.debug(f"[TIMING] Agent stopped speaking at {timestamp:.3f}")

            # Mark that agent stopped speaking
            self.metrics.agent_bot_is_speaking = False

            # Finalize the turn with accumulated transcript
            if self.metrics.agent_current_transcript:
                complete_text = self.metrics.agent_current_transcript.strip()
                logger.info(f"[AGENT] {complete_text}")

                # Match transcript to audio segment
                protocol_time = self._get_relative_time(timestamp)
                self._match_transcript_to_audio_segment("agent", complete_text, protocol_time)

                # Update the last latency measurement with complete agent transcript
                if self.metrics.latencies and not self.metrics.latencies[-1].agent_transcript:
                    self.metrics.latencies[-1].agent_transcript = complete_text

                turn_data = {
                    "timestamp": datetime.now().isoformat(),
                    "role": "agent",
                    "text": complete_text,
                }
                self.metrics.turns.append(turn_data)

                # Find the matched audio segment for accurate timing
                matched_audio_segment = None
                for seg in self.metrics.audio_segments:
                    if seg.speaker == "agent" and seg.transcript == complete_text:
                        matched_audio_segment = seg
                        break

                # If no match in completed segments, check in-progress segment
                if not matched_audio_segment and self.metrics.audio_agent_segment_in_progress:
                    if self.metrics.audio_agent_segment_in_progress.transcript == complete_text:
                        matched_audio_segment = self.metrics.audio_agent_segment_in_progress

                # Use audio segment timing if available, otherwise fallback to protocol timing
                if matched_audio_segment:
                    segment_start = matched_audio_segment.start_time
                    segment_end = matched_audio_segment.end_time
                else:
                    # Fallback to protocol-based timing from current_agent_segment
                    if self.metrics.current_agent_segment is not None:
                        segment_start = self.metrics.current_agent_segment.start_time
                        segment_end = self._get_relative_time(timestamp)
                    else:
                        segment_start = 0.0
                        segment_end = self._get_relative_time(timestamp)

                # Calculate latency based on audio segment timing
                latency_ms = None
                if matched_audio_segment and self.metrics.audio_segments:
                    # Find the most recent user segment before this agent segment
                    for seg in reversed(self.metrics.audio_segments):
                        if seg.speaker == "user" and seg.end_time < matched_audio_segment.start_time:
                            latency_ms = (matched_audio_segment.start_time - seg.end_time) * 1000
                            break

                if self.log_file:
                    log_entry = self._format_turn_log("agent", complete_text, segment_start, segment_end, latency_ms)
                    self.metrics.log_entries.append((segment_start, log_entry))

                # Create segment entry for segLST
                if self.audio_file and self.metrics.current_agent_segment is not None:
                    # Finalize the current segment with transcript and end time
                    if self.metrics.current_agent_segment is not None:
                        self.metrics.current_agent_segment.end_time = segment_end
                        self.metrics.current_agent_segment.transcript = complete_text
                        self.metrics.segments.append(self.metrics.current_agent_segment)
                        self.metrics.current_agent_segment = None
                    else:
                        # Fallback: create segment without audio chunks (shouldn't happen normally)
                        segment = SegmentEntry(
                            start_time=segment_start, end_time=segment_end, speaker="agent", transcript=complete_text
                        )
                        self.metrics.segments.append(segment)

                # Clear accumulated text for next turn
                self.metrics.agent_current_transcript = ""
            else:
                # No transcript accumulated - response was likely interrupted
                # Create placeholder entry since audio was still recorded
                logger.warning(f"[AGENT] Stopped speaking but no transcript was accumulated (interrupted)")
                complete_text = "[turn interrupted]"
                logger.info(f"[AGENT] {complete_text}")

                # Check if this is the first agent turn (skip initial empty turn on connection)
                has_previous_agent_segments = any(seg.speaker == "agent" for seg in self.metrics.segments)

                if has_previous_agent_segments:
                    # Only log interrupted turns after the first one
                    # Update the last latency measurement with interrupted placeholder
                    if self.metrics.latencies and not self.metrics.latencies[-1].agent_transcript:
                        self.metrics.latencies[-1].agent_transcript = complete_text

                    turn_data = {
                        "timestamp": datetime.now().isoformat(),
                        "role": "agent",
                        "text": complete_text,
                    }
                    self.metrics.turns.append(turn_data)

                    # Calculate turn timing for log and segLST from current_agent_segment
                    if self.metrics.current_agent_segment is not None:
                        segment_start = self.metrics.current_agent_segment.start_time
                        segment_end = self._get_relative_time(timestamp)
                    else:
                        segment_start = 0.0
                        segment_end = self._get_relative_time(timestamp)

                    if self.log_file:
                        log_entry = self._format_turn_log("agent", complete_text, segment_start, segment_end)
                        self.metrics.log_entries.append((segment_start, log_entry))

                    # Create segment entry for segLST with interrupted placeholder
                    if self.audio_file and self.metrics.current_agent_segment is not None:
                        # Finalize the current segment with interrupted text
                        self.metrics.current_agent_segment.end_time = segment_end
                        self.metrics.current_agent_segment.transcript = complete_text
                        self.metrics.segments.append(self.metrics.current_agent_segment)
                        self.metrics.current_agent_segment = None
                else:
                    # This is the first agent turn - just clean up without logging
                    logger.info(f"[AGENT] Skipping first empty/interrupted turn on connection")
                    if self.metrics.current_agent_segment is not None:
                        self.metrics.current_agent_segment = None

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

    def _save_audio_and_seglst(self):
        """Save stereo audio file and segLST transcript file."""

        if not self.audio_file:
            logger.warning("[DEBUG] No audio file to save")
            return

        try:
            logger.info(f"Saving audio to {self.audio_file}...")

            # Use audio-based segments for accurate timing
            segments_to_save = self.metrics.audio_segments if self.metrics.audio_segments else self.metrics.segments

            # Determine total duration from segments
            if not segments_to_save:
                logger.warning("No segments to save")
                return

            max_end_time = max(seg.end_time for seg in segments_to_save)
            total_samples = int(np.ceil(max_end_time * self.output_sample_rate))

            # Create silent stereo buffer
            stereo_audio = np.zeros((total_samples, 2), dtype=np.int16)

            # Place each segment's audio at the correct timestamp
            for seg in segments_to_save:
                if not seg.audio_chunks:
                    logger.debug(f"Segment has no audio: {seg.speaker} at {seg.start_time:.3f}s")
                    continue

                # Determine which sample rate the audio chunks are actually at
                # Audio chunks are stored after AudioStream resampling:
                # - User audio: resampled to agent_input_sample_rate (for USER→AGENT stream)
                # - Agent audio: resampled to user_input_sample_rate (for AGENT→USER stream)
                if seg.speaker == "user":
                    source_sample_rate = self.agent_input_sample_rate
                    channel_idx = 0  # Left channel
                else:  # agent
                    source_sample_rate = self.user_input_sample_rate
                    channel_idx = 1  # Right channel

                # Resample segment audio to output sample rate
                segment_audio = self._resample_audio_for_saving(
                    seg.audio_chunks, source_sample_rate, self.output_sample_rate
                )

                # Calculate start sample position
                start_sample = int(seg.start_time * self.output_sample_rate)

                # Calculate how many samples to place (don't exceed segment duration)
                segment_duration_samples = int((seg.end_time - seg.start_time) * self.output_sample_rate)
                samples_to_place = min(len(segment_audio), segment_duration_samples)

                # Ensure we don't write beyond the buffer
                end_sample = min(start_sample + samples_to_place, total_samples)
                actual_samples = end_sample - start_sample

                # Place audio in the correct channel at the correct position
                stereo_audio[start_sample:end_sample, channel_idx] = segment_audio[:actual_samples]

                logger.debug(
                    f"Placed {seg.speaker} audio: {start_sample}-{end_sample} samples ({actual_samples} samples, {actual_samples / self.output_sample_rate:.3f}s)"
                )

            # Save as WAV file
            with wave.open(str(self.audio_file), 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.output_sample_rate)
                wav_file.writeframes(stereo_audio.tobytes())

            logger.info(f"Audio saved: {self.audio_file}")
            logger.info(f"  Channels: 2 (user=left, agent=right)")
            logger.info(f"  Sample rate: {self.output_sample_rate} Hz")
            logger.info(f"  Duration: {total_samples / self.output_sample_rate:.2f}s")

            # Save segLST file in JSON format

            # Prepare JSON data
            session_id = self.scenario_name or Path(self.audio_file).stem

            segments_json = []
            sorted_segments = sorted(segments_to_save, key=lambda s: s.start_time)
            for seg in sorted_segments:
                # Apply offsets but ensure end_time remains after start_time
                start_with_offset = seg.start_time + self.seglst_start_offset_seconds
                end_with_offset = seg.end_time + self.seglst_end_offset_seconds

                # Validate: if offsets cause negative duration, skip offsets for this segment
                if end_with_offset <= start_with_offset:
                    logger.warning(
                        f"[SEGLST] Offsets would create negative duration for {seg.speaker} segment "
                        f"(original: {seg.start_time:.3f}-{seg.end_time:.3f}), skipping offsets"
                    )
                    start_with_offset = seg.start_time
                    end_with_offset = seg.end_time

                segments_json.append(
                    {
                        "session_id": session_id,
                        "words": seg.transcript,
                        "speaker": seg.speaker,
                        "start_time": start_with_offset,
                        "end_time": end_with_offset,
                    }
                )

            # Write JSON file
            with open(self.seglst_file, 'w') as f:
                json.dump(segments_json, f, indent=2)

            logger.info(f"segLST saved: {self.seglst_file}")
            logger.info(f"  Total segments: {len(segments_to_save)}")
            if self.metrics.audio_segments:
                logger.info(f"  (Using audio-based segments for accurate timing)")

        except Exception as e:
            logger.error(f"Error saving audio/segLST: {e}")
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
