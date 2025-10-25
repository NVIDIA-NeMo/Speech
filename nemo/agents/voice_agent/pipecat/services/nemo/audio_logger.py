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

import json
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger


class AudioLogger:
    """
    Utility class for logging audio data and transcriptions during voice agent interactions.

    This logger saves:
    - Audio files in WAV format
    - Transcriptions with metadata in JSON format
    - Session information and metadata

    File structure:
        log_dir/
        ├── session_YYYYMMDD_HHMMSS/
        │   ├── user/
        │   │   ├── 00001_HHMMSS.wav
        │   │   ├── 00001_HHMMSS.json
        │   │   ├── 00002_HHMMSS.wav
        │   │   └── 00002_HHMMSS.json
        │   ├── agent/
        │   │   ├── 00001_HHMMSS.wav
        │   │   ├── 00001_HHMMSS.json
        │   └── session_metadata.json

    Args:
        log_dir: Base directory for storing logs (default: "./audio_logs")
        session_id: Optional custom session ID. If None, auto-generated from timestamp
        enabled: Whether logging is enabled (default: True)
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "./audio_logs",
        session_id: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        if not self.enabled:
            logger.info("AudioLogger is disabled")
            return

        self.log_dir = Path(log_dir)

        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session_id = session_id
        self.session_dir = self.log_dir / session_id

        # Create directories
        self.user_dir = self.session_dir / "user"
        self.agent_dir = self.session_dir / "agent"

        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        # Counters for file naming (thread-safe)
        self._user_counter = 0
        self._agent_counter = 0
        self._lock = threading.Lock()
        self._staged_metadata = None
        self._staged_audio_data = None

        # Session metadata
        self.session_metadata = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "user_entries": [],
            "agent_entries": [],
        }

        logger.info(f"AudioLogger initialized: {self.session_dir}")

    def _get_next_counter(self, speaker: str) -> int:
        """Get the next counter value for a speaker in a thread-safe manner."""
        with self._lock:
            if speaker == "user":
                self._user_counter += 1
                return self._user_counter
            else:
                self._agent_counter += 1
                return self._agent_counter

    def _save_audio_wav(
        self,
        audio_data: Union[bytes, np.ndarray],
        file_path: Path,
        sample_rate: int,
        num_channels: int = 1,
    ):
        """
        Save audio data to a WAV file.

        Args:
            audio_data: Audio data as bytes or numpy array
            file_path: Path to save the WAV file
            sample_rate: Audio sample rate in Hz
            num_channels: Number of audio channels (default: 1)
        """
        try:
            # Convert audio data to bytes if it's a numpy array
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype in [np.float32, np.float64]:
                    # Convert float [-1, 1] to int16 [-32768, 32767]
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype != np.int16:
                    audio_data = audio_data.astype(np.int16)
                audio_bytes = audio_data.tobytes()
            else:
                audio_bytes = audio_data

            # Write WAV file
            with wave.open(str(file_path), 'wb') as wav_file:
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)

            logger.debug(f"Saved audio to {file_path}")
        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {e}")
            raise

    def _save_metadata_json(self, metadata: dict, file_path: Path):
        """Save metadata to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved metadata to {file_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to {file_path}: {e}")
            raise

    def stage_user_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        transcription: str,
        sample_rate: int = 16000,
        num_channels: int = 1,
        is_final: bool = True,
        additional_metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Stage log user audio and transcription (from STT).
        This data will be saved when the turn is complete by `log_user_audio` method.

        Args:
            audio_data: Raw audio data as bytes or numpy array
            transcription: Transcribed text
            sample_rate: Audio sample rate in Hz (default: 16000)
            num_channels: Number of audio channels (default: 1)
            is_final: Whether this is a final transcription (default: True)
            additional_metadata: Additional metadata to include

        Returns:
            Dictionary with logged file paths, or None if logging is disabled
        """
        if not self.enabled:
            return None

        try:
            # Get counter and generate filenames
            counter = self._get_next_counter("user")
            timestamp = datetime.now().strftime('%H%M%S')
            base_name = f"{counter:05d}_{timestamp}"

            audio_file = self.user_dir / f"{base_name}.wav"
            metadata_file = self.user_dir / f"{base_name}.json"

            # Save audio
            # self._save_audio_wav(audio_data, audio_file, sample_rate, num_channels)
            self._staged_audio_data = audio_data

            # Prepare metadata
            self._staged_metadata = {
                "base_name": base_name,
                "counter": counter,
                "speaker": "user",
                "timestamp": datetime.now().isoformat(),
                "transcription": transcription,
                "is_final": is_final,
                "audio_file": audio_file.name,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "audio_duration_sec": (
                    len(audio_data) / (sample_rate * num_channels * 2)
                    if isinstance(audio_data, bytes)
                    else len(audio_data) / sample_rate
                ),
            }

            if additional_metadata:
                self._staged_metadata.update(additional_metadata)

            # Save metadata
            # self._save_metadata_json(metadata, metadata_file)

            # logger.info(f"Logged user audio #{counter}: '{transcription[:50]}{'...' if len(transcription) > 50 else ''}'")

            return {
                "audio_file": str(audio_file),
                "metadata_file": str(metadata_file),
                "counter": counter,
            }

        except Exception as e:
            logger.error(f"Error logging user audio: {e}")
            return None

    def save_user_audio(self):
        """Save the user audio to the disk."""
        audio_file = self.user_dir / f"{self._staged_metadata['base_name']}.wav"
        metadata_file = self.user_dir / f"{self._staged_metadata['base_name']}.json"

        self._save_audio_wav(
            audio_data=self._staged_audio_data, file_path=audio_file, sample_rate=self._staged_metadata["sample_rate"]
        )

        self._save_metadata_json(metadata=self._staged_metadata, file_path=metadata_file)
        logger.info(
            f"Saved user audio #{self._staged_metadata['counter']}: '{self._staged_metadata['transcription'][:50]}{'...' if len(self._staged_metadata['transcription']) > 50 else ''}'"
        )
        # Update session metadata
        with self._lock:
            self.session_metadata["user_entries"].append(self._staged_metadata)
            self._save_session_metadata()

    def log_agent_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        text: str,
        sample_rate: int = 22050,
        num_channels: int = 1,
        additional_metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Log agent audio and text (from TTS).

        Args:
            audio_data: Generated audio data as bytes or numpy array
            text: Input text that was synthesized
            sample_rate: Audio sample rate in Hz (default: 22050)
            num_channels: Number of audio channels (default: 1)
            additional_metadata: Additional metadata to include

        Returns:
            Dictionary with logged file paths, or None if logging is disabled
        """
        if not self.enabled:
            return None

        try:
            # Get counter and generate filenames
            counter = self._get_next_counter("agent")
            timestamp = datetime.now().strftime('%H%M%S')
            base_name = f"{counter:05d}_{timestamp}"

            audio_file = self.agent_dir / f"{base_name}.wav"
            metadata_file = self.agent_dir / f"{base_name}.json"

            # Save audio
            self._save_audio_wav(audio_data, audio_file, sample_rate, num_channels)

            # Prepare metadata
            metadata = {
                "counter": counter,
                "speaker": "agent",
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "audio_file": audio_file.name,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "audio_duration_sec": (
                    len(audio_data) / (sample_rate * num_channels * 2)
                    if isinstance(audio_data, bytes)
                    else len(audio_data) / sample_rate
                ),
            }

            if additional_metadata:
                metadata.update(additional_metadata)

            # Save metadata
            self._save_metadata_json(metadata, metadata_file)

            # Update session metadata
            with self._lock:
                self.session_metadata["agent_entries"].append(metadata)
                self._save_session_metadata()

            logger.info(f"Logged agent audio #{counter}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            return {
                "audio_file": str(audio_file),
                "metadata_file": str(metadata_file),
                "counter": counter,
            }

        except Exception as e:
            logger.error(f"Error logging agent audio: {e}")
            return None

    def _save_session_metadata(self):
        """Save the session metadata to disk."""
        if not self.enabled:
            return

        try:
            metadata_file = self.session_dir / "session_metadata.json"
            self.session_metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata_json(self.session_metadata, metadata_file)
        except Exception as e:
            logger.error(f"Error saving session metadata: {e}")

    def finalize_session(self):
        """Finalize the session and save final metadata."""
        if not self.enabled:
            return

        self.session_metadata["end_time"] = datetime.now().isoformat()
        self.session_metadata["total_user_entries"] = self._user_counter
        self.session_metadata["total_agent_entries"] = self._agent_counter
        self._save_session_metadata()
        logger.info(f"Session finalized: {self.session_id} (User: {self._user_counter}, Agent: {self._agent_counter})")

    def get_session_info(self) -> dict:
        """Get current session information."""
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "user_entries": self._user_counter,
            "agent_entries": self._agent_counter,
            "enabled": self.enabled,
        }
