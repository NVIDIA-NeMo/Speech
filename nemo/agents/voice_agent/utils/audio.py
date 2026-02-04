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
import time
from queue import Queue

import numpy as np
import soxr
from loguru import logger

STREAM_TIMEOUT_SECS = 0.2


class SOXRAudioResampler:
    """
    An audio resampler that uses the SoX resampler library. It's stateless and will return the result immediately.
    """

    def __init__(self, in_sample_rate: int, out_sample_rate: int, quality: str = "VHQ", *args, **kwargs):
        """Initialize the SoX audio resampler.

        Args:
            in_sample_rate: The sample rate of the input audio.
            out_sample_rate: The sample rate of the output audio.
            quality: The quality of the resampling.
            **kwargs: Additional keyword arguments (currently unused).
        """
        self.quality = quality
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate

    def resample(self, audio: bytes) -> bytes:
        """Resample audio data using SoX resampler library.

        Args:
            audio: Input audio data as raw bytes (16-bit signed integers).

        Returns:
            Resampled audio data as raw bytes (16-bit signed integers).
        """
        if self.in_sample_rate == self.out_sample_rate:
            return audio
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = soxr.resample(audio_data, self.in_sample_rate, self.out_sample_rate, quality=self.quality)
        result = resampled_audio.astype(np.int16).tobytes()
        return result


class SOXRAudioStreamResampler:
    """
    A class that resamples an audio stream using the SoX resampler library.
    """

    def __init__(self, in_sample_rate: int, out_sample_rate: int, quality: str = "VHQ", *args, **kwargs):
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.quality = quality
        self.resampler = soxr.ResampleStream(
            in_sample_rate, out_sample_rate, quality=quality, num_channels=1, dtype="int16"
        )
        self._last_resample_time = None

    def _should_flush(self):
        """
        Check if the resampler should be flushed.
        """
        if self._last_resample_time is None:
            return False
        return time.time() - self._last_resample_time > STREAM_TIMEOUT_SECS

    def reset(self):
        """
        Reset the resampler.
        """
        self._last_resample_time = None
        self.resampler.clear()

    def resample(self, audio: bytes):
        """
        Resample an audio chunk using the SoX resampler library.
        Args:
            audio: The audio chunk to resample.
        Returns:
            The resampled audio chunk.
        """
        is_last = self._should_flush()
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = self.resampler.resample_chunk(audio_data, last=is_last)
        self._last_resample_time = time.time()
        if is_last:
            self.reset()
        result = resampled_audio.astype(np.int16).tobytes()
        return result


class AudioStream:
    """
    A class that simulates a realtime audio stream. It caches the input audio chunks
    and resamples them to the output sample rate. Each time its get() function is called,
    it returns the next chunk of audio at the output sample rate. If the audio cache doesn't
    have enough audio to fill the output chunk, it will append silence to the output chunk.

    The class will be used in an asyncio context, where one thread is putting audio chunks
    into the cache and another thread is getting audio chunks from the cache.
    """

    def __init__(
        self,
        chunk_size_in_seconds: float,
        input_sample_rate: int,
        output_sample_rate: int,
        stream_resampler: bool = True,
        tag: str = "",
        min_buffer_chunks: int = 10,
        drain_threshold: int = 5,
        min_sustain_chunks: int = 1,
    ):
        self.chunk_size_in_seconds = chunk_size_in_seconds
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.stream_resampler = stream_resampler
        self.output_chunk_bytes = int(self.output_sample_rate * self.chunk_size_in_seconds) * 2  # 16-bit audio
        self.tag = tag
        self.min_buffer_chunks = min_buffer_chunks
        self._buffer_ready = False
        # Initialize the appropriate resampler
        if self.stream_resampler:
            self.resampler = SOXRAudioStreamResampler(input_sample_rate, output_sample_rate, quality="VHQ")
        else:
            self.resampler = SOXRAudioResampler(input_sample_rate, output_sample_rate, quality="VHQ")

        # Use asyncio.Queue for async/await compatibility
        self.audio_cache = asyncio.Queue()

        # Buffer for partial chunks
        self.output_buffer = b''

        self._buffer_empty_count = 0  # Track consecutive empty returns
        self.drain_threshold = drain_threshold  # Only reset ready after 5 consecutive underflows (~80ms of silence)
        self.min_sustain_chunks = min_sustain_chunks
        self._next_send_time = 0

    async def put(self, audio_chunk: bytes):
        """
        Put an audio chunk into the audio cache after resampling.

        Args:
            audio_chunk: Input audio chunk at input_sample_rate
        """
        # Resample the audio chunk to output sample rate
        await self.audio_cache.put(audio_chunk)
        audio_len_in_seconds = len(audio_chunk) / 2 / self.input_sample_rate
        logger.debug(
            f"[{self.tag}] Put {len(audio_chunk)} bytes ({audio_len_in_seconds:.4f} seconds) into AudioStream"
        )

    def resample(self, audio_chunk: bytes) -> bytes:
        """
        Resample an audio chunk from input sample rate to output sample rate.

        Args:
            audio_chunk: Raw audio bytes (16-bit signed integers)

        Returns:
            Resampled audio bytes (16-bit signed integers)
        """
        if self.input_sample_rate == self.output_sample_rate:
            return audio_chunk

        return self.resampler.resample(audio_chunk)

    def maybe_pad_silence(self, audio_chunk: bytes) -> bytes:
        """
        Pad audio chunk with silence if it's shorter than the expected output chunk size.

        Args:
            audio_chunk: Audio bytes (16-bit signed integers)

        Returns:
            Audio chunk padded to output_chunk_bytes with silence (zeros)
        """
        current_length = len(audio_chunk)

        if current_length >= self.output_chunk_bytes:
            return audio_chunk

        # Calculate padding needed
        padding_bytes = self.output_chunk_bytes - current_length

        # Pad with zeros (silence for 16-bit audio)
        padded_chunk = audio_chunk + (b'\x00' * padding_bytes)

        return padded_chunk

    @property
    def current_buffer_size(self) -> int:
        """
        Get the current size of the buffer.
        """
        return len(self.output_buffer) // self.output_chunk_bytes

    def _is_buffer_full(self) -> bool:
        """
        Check if the buffer is full.
        """
        return self.current_buffer_size >= self.min_buffer_chunks

    async def _send_audio_sleep(self):
        """Simulate audio device timing by sleeping between audio chunks."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self.chunk_size_in_seconds
        else:
            self._next_send_time += self.chunk_size_in_seconds

    async def get_nowait(self) -> bytes:
        """
        Get the next output chunk of audio, immediately padding with silence if no audio is available.
        """
        return await self.get_wait(no_wait=True)

    async def get_wait(self, timeout: float = None, no_wait: bool = False) -> bytes:
        """
        Get the next output chunk of audio, WAITING for audio to be available.

        Unlike get(), this method will block and wait for audio to arrive rather than
        immediately padding with silence. This prevents gaps in audio when packets
        arrive in bursts (common in WebSocket/network scenarios).

        Use this for continuous audio streaming where you want smooth audio without
        artificial gaps.

        Args:
            timeout: Maximum time to wait in seconds (None = no wait)
            no_wait: If True, only tries to read the audio cache once, and returns silence
                    immediately if no audio is available.
        Returns:
            Audio chunk of exactly output_chunk_bytes (16-bit signed integers)
        """
        start_time = time.time()
        if no_wait:
            timeout = None
        while True:
            try:
                # Calculate remaining time budget BEFORE waiting
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = timeout - elapsed
                    if remaining_timeout <= 0:
                        break  # Out of time budget
                else:
                    remaining_timeout = None
                if remaining_timeout is not None:
                    chunk = await asyncio.wait_for(self.audio_cache.get(), timeout=remaining_timeout)
                else:
                    chunk = self.audio_cache.get_nowait()
                chunk = self.resample(chunk)
                self.output_buffer += chunk
                logger.debug(
                    f"[{self.tag}] Added {len(chunk)} bytes ({len(chunk) / 2 / self.output_sample_rate:.4f} seconds) to buffer, current buffer size: {self.current_buffer_size}"
                )
                if self._is_buffer_full() or no_wait:
                    break
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break

        if self._is_buffer_full():
            self._buffer_ready = True

        # Check if buffer too low to sustain
        if self._buffer_ready and self.current_buffer_size < self.min_sustain_chunks:
            # Only reset if we've been low for a while
            self._buffer_empty_count += 1
            if self._buffer_empty_count > self.drain_threshold:
                self._buffer_ready = False
                logger.warning(
                    f"[{self.tag}] Buffer sustained low, resetting (empty count: {self._buffer_empty_count})"
                )
        else:
            self._buffer_empty_count = 0

        if not self._buffer_ready:
            logger.debug(
                f"[{self.tag}] Buffer not ready ({self.current_buffer_size}/{self.min_buffer_chunks} chunks), sending silence"
            )
            return self.maybe_pad_silence(b'')

        logger.debug(
            f"[{self.tag}] Buffer ready ({self.current_buffer_size}/{self.min_buffer_chunks} chunks), sending audio"
        )
        output_chunk = self.output_buffer
        # If we have more than needed, split it
        if len(output_chunk) > self.output_chunk_bytes:
            result = output_chunk[: self.output_chunk_bytes]
            self.output_buffer = output_chunk[self.output_chunk_bytes :]
            return result
        elif len(output_chunk) == self.output_chunk_bytes:
            # Exactly the right amount
            self.output_buffer = b''
            return output_chunk
        else:
            # Send a silence chunk when buffer is less than one chunk
            self._buffer_empty_count += 1
            # Only reset ready after 5 consecutive underflows (~80ms of silence)
            if self._buffer_empty_count > self.drain_threshold:
                self._buffer_ready = False
                logger.warning(f"[{self.tag}] Buffer drained, resetting (empty count: {self._buffer_empty_count})")
            logger.debug(f"[{self.tag}] Buffer empty, sending silence (empty count: {self._buffer_empty_count})")
            return self.maybe_pad_silence(b'')
