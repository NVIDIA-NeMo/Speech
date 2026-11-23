# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


from typing import Iterator

import torch

from nemo.collections.asr.inference.streaming.framing.mono_stream import MonoStream
from nemo.collections.asr.inference.streaming.framing.request import Frame, RequestOptions
from nemo.collections.asr.inference.streaming.framing.stream import Stream
from nemo.collections.asr.inference.utils.progressbar import ProgressBar


class MultiStream:
    """MultiStreamer for multiple streams"""

    def __init__(self, n_frames_per_stream: int):
        """
        Args:
            n_frames_per_stream (int): Number of frames per stream
        """
        self.n_frames_per_stream = n_frames_per_stream
        self.streams = {}

    def add_stream(self, stream: Stream, stream_id: int) -> None:
        """
        Add a stream to the streamer
        Args:
            stream (Stream): The stream to add
            stream_id (int): The id of the stream
        """
        self.streams[stream_id] = iter(stream)

    def rm_stream(self, stream_id: int) -> None:
        """
        Remove a stream from the streamer
        Args:
            stream_id (int): The id of the stream
        """
        self.streams.pop(stream_id, None)

    def __len__(self) -> int:
        """Number of running streams"""
        return len(self.streams)

    def __iter__(self) -> Iterator:
        """Returns the iterator object"""
        return self

    def __next__(self) -> list[Frame]:
        """
        Get the next batch of frames
        Returns:
            list[Frame]: The next batch of frames
        """
        frame_batch = []
        ids_to_remove = []
        for stream_id, stream_iter in self.streams.items():
            # Get n_frames_per_stream frames from each stream
            for _ in range(self.n_frames_per_stream):
                frame = next(stream_iter)[0]
                frame_batch.append(frame)
                if frame.is_last:
                    ids_to_remove.append(stream_id)

        # Remove streams that have ended
        for stream_id in ids_to_remove:
            self.rm_stream(stream_id)

        # If no frames are generated, raise StopIteration
        if len(frame_batch) == 0:
            raise StopIteration

        return frame_batch


class ContinuousBatchedFrameStreamer:
    """
    A class that manages continuous streaming of audio frames from multiple audio files, providing
    frame generation in batches. The class supports dynamically adding audio streams, updating
    a progress bar, and yielding batches of frames for further processing.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_size_in_secs: float,
        batch_size: int,
        n_frames_per_stream: int,
        pad_last_frame: bool = False,
        flush_size_in_secs: float = 0.0,
    ):
        """
        Args:
            sample_rate (int): The sample rate of the audio
            frame_size_in_secs (float): The size of the frame in seconds
            batch_size (int): The batch size
            n_frames_per_stream (int): The number of frames per stream
            pad_last_frame (bool): Whether to pad the last frame
            flush_size_in_secs (float): Seconds of silence appended to every stream, 0.0 for none
        """

        self.sample_rate = sample_rate
        self.frame_size_in_secs = frame_size_in_secs
        self.batch_size = batch_size
        self.pad_last_frame = pad_last_frame
        self.flush_size_in_secs = flush_size_in_secs

        self.multi_streamer = MultiStream(n_frames_per_stream=n_frames_per_stream)
        self.stream_id = 0

        self._progress_bar = None
        self.processed_streams = set()

    def set_audio_filepaths(
        self,
        audio_filepaths: list[str],
        options: list[RequestOptions],
        audio_samples: list[torch.Tensor] | None = None,
    ) -> None:
        """
        Set the audio filepaths
        Args:
            audio_filepaths (list[str]): The list of audio filepaths
            options (list[RequestOptions]): The list of options
            audio_samples (list[torch.Tensor] | None): Preloaded audio samples, one tensor per filepath.
                When given, streams are fed from these tensors and no audio file is read from disk.
        """
        if len(audio_filepaths) != len(options):
            raise ValueError("audio_filepaths and options must have the same length")
        if audio_samples is not None and len(audio_samples) != len(audio_filepaths):
            raise ValueError("audio_filepaths and audio_samples must have the same length")

        self.audio_filepaths = audio_filepaths
        self.options = options
        self.audio_samples = audio_samples
        self.n_audio_files = len(audio_filepaths)
        self.total_progress_steps = self.n_audio_files * 2  # One step for adding, one for processing
        self.sid2filepath = {}
        self.elapsed_durations = {}

    def set_progress_bar(self, progress_bar: ProgressBar) -> None:
        """
        Set the progress bar
        Args:
            progress_bar (ProgressBar): The progress bar to set
        """
        self._progress_bar = progress_bar
        self.restart_progress_bar()

    def restart_progress_bar(self) -> None:
        """Restart the progress bar"""
        if self._progress_bar:
            self._progress_bar.restart()

    def update_progress_bar(self) -> None:
        """Update the progress bar"""
        if self._progress_bar:
            self._progress_bar.update_bar(1 / self.total_progress_steps)

    def finish_progress_bar(self) -> None:
        """Finish the progress bar"""
        if self._progress_bar:
            self._progress_bar.finish()

    def get_audio_filepath(self, stream_id: int) -> str:
        """
        Get the audio filepath for a given stream id
        Args:
            stream_id (int): The id of the stream
        Returns:
            str: The audio filepath for the given stream id
        """
        return self.sid2filepath[stream_id]

    def get_elapsed_duration(self, stream_id: int) -> float:
        """
        Get the elapsed audio duration for a given stream id
        Args:
            stream_id (int): The id of the stream
        Returns:
            float: The elapsed audio duration for the given stream id
        """
        return self.elapsed_durations[stream_id]

    def __iter__(self) -> Iterator:
        """Returns the iterator object"""
        return self

    def add_stream(self) -> None:
        """Create a new stream and add it to the streamer"""
        if self.stream_id >= self.n_audio_files:
            return  # No more files to add

        # Create a new stream
        stream = MonoStream(
            self.sample_rate,
            self.frame_size_in_secs,
            stream_id=self.stream_id,
            pad_last_frame=self.pad_last_frame,
            flush_size_in_secs=self.flush_size_in_secs,
        )
        # Load the next audio file
        audio_filepath = self.audio_filepaths[self.stream_id]
        options = self.options[self.stream_id]
        self.sid2filepath[self.stream_id] = audio_filepath
        self.elapsed_durations[self.stream_id] = 0.0
        if self.audio_samples is not None:
            stream.load_audio(self.audio_samples[self.stream_id], options)
        else:
            stream.load_audio(audio_filepath, options)

        # Add the stream to the multi streamer
        self.multi_streamer.add_stream(stream, stream_id=self.stream_id)
        self.stream_id += 1

        # Update the progress bar
        self.update_progress_bar()

    def __next__(self) -> list[Frame]:
        """
        Get the next batch of frames, continuously adding streams
        Returns:
            list[Frame]: The next batch of frames
        """
        # If there are fewer streams than batch size, add more streams
        while len(self.multi_streamer) < self.batch_size and self.stream_id < self.n_audio_files:
            self.add_stream()

        try:
            frames = next(self.multi_streamer)
            # Update progress when a stream is fully processed
            for frame in frames:
                sid = frame.stream_id
                self.elapsed_durations[sid] += frame.valid_size / self.sample_rate
                if sid not in self.processed_streams and frame.is_last:
                    self.processed_streams.add(sid)
                    self.update_progress_bar()
            return frames
        except StopIteration:
            # if there are remaining streams, add them
            if self.stream_id < self.n_audio_files:
                return self.__next__()

        if self.stream_id == self.n_audio_files:
            self.finish_progress_bar()
            raise StopIteration

        raise ValueError("stream_id > self.n_audio_files unexpected")
