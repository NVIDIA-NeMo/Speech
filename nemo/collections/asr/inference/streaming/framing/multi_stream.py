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


from typing import Callable, Iterator

import torch

from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.streaming.framing.mono_stream import MonoStream
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request, RequestOptions
from nemo.collections.asr.inference.streaming.framing.stream import Stream
from nemo.collections.asr.inference.utils.enums import RequestType
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
    ):
        """
        Args:
            sample_rate (int): The sample rate of the audio
            frame_size_in_secs (float): The size of the frame in seconds
            batch_size (int): The batch size
            n_frames_per_stream (int): The number of frames per stream
            pad_last_frame (bool): Whether to pad the last frame
        """

        self.sample_rate = sample_rate
        self.frame_size_in_secs = frame_size_in_secs
        self.batch_size = batch_size
        self.pad_last_frame = pad_last_frame

        self.multi_streamer = MultiStream(n_frames_per_stream=n_frames_per_stream)
        self.stream_id = 0

        self._progress_bar = None
        self.processed_streams = set()

    def set_audio_filepaths(self, audio_filepaths: list[str], options: list[RequestOptions]) -> None:
        """
        Set the audio filepaths
        Args:
            audio_filepaths (list[str]): The list of audio filepaths
            options (list[RequestOptions]): The list of options
        """
        if len(audio_filepaths) != len(options):
            raise ValueError("audio_filepaths and options must have the same length")

        self.audio_filepaths = audio_filepaths
        self.options = options
        self.n_audio_files = len(audio_filepaths)
        self.total_progress_steps = self.n_audio_files * 2  # One step for adding, one for processing
        self.sid2filepath = {}

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

    def __iter__(self) -> Iterator:
        """Returns the iterator object"""
        return self

    def add_stream(self) -> None:
        """Create a new stream and add it to the streamer"""
        if self.stream_id >= self.n_audio_files:
            return  # No more files to add

        # Create a new stream
        stream = MonoStream(
            self.sample_rate, self.frame_size_in_secs, stream_id=self.stream_id, pad_last_frame=self.pad_last_frame
        )
        # Load the next audio file
        audio_filepath = self.audio_filepaths[self.stream_id]
        options = self.options[self.stream_id]
        self.sid2filepath[self.stream_id] = audio_filepath
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
                if frame.stream_id not in self.processed_streams and frame.is_last:
                    self.processed_streams.add(frame.stream_id)
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


class ContinuousBatchedRequestStreamer:
    """
    A class that manages continuous streaming of requests from multiple audio files, providing
    request generation in batches. Requests can be frames or feature buffers.
    The class supports dynamically adding audio streams, updating a progress bar,
    and yielding batches of requests for further processing.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_size_in_secs: float,
        batch_size: int,
        n_frames_per_stream: int,
        request_type: RequestType = RequestType.FRAME,
        preprocessor: Callable = None,
        buffer_size_in_secs: float = None,
        device: torch.device = None,
        pad_last_frame: bool = False,
        right_pad_features: bool = False,
        tail_padding_in_samples: int = 0,
    ):
        """
        Args:
            sample_rate (int): The sample rate of the audio
            frame_size_in_secs (float): The size of the frame in seconds
            batch_size (int): The batch size
            n_frames_per_stream (int): The number of frames per stream
            request_type (RequestType): The type of request
            preprocessor (Callable): Preprocessor object, required for request type FEATURE_BUFFER
            buffer_size_in_secs (float): The size of the buffer in seconds, required for request type FEATURE_BUFFER
            device (torch.device): The device to use, required for request type FEATURE_BUFFER
            pad_last_frame (bool): Whether to pad the last frame
            right_pad_features (bool): Whether to right pad the features, optional for request type FEATURE_BUFFER
            tail_padding_in_samples (int): The tail padding in samples, optional for request type FEATURE_BUFFER
        """

        if request_type is RequestType.FEATURE_BUFFER:
            if buffer_size_in_secs is None:
                raise ValueError("buffer_size_in_secs must be provided for request type FEATURE_BUFFER")
            if preprocessor is None:
                raise ValueError("preprocessor must be provided for request type FEATURE_BUFFER")
            if device is None:
                raise ValueError("device must be provided for request type FEATURE_BUFFER")

        self.request_type = request_type
        self.multi_streamer = ContinuousBatchedFrameStreamer(
            sample_rate=sample_rate,
            frame_size_in_secs=frame_size_in_secs,
            batch_size=batch_size,
            n_frames_per_stream=n_frames_per_stream,
            pad_last_frame=pad_last_frame,
        )

        if self.request_type is RequestType.FEATURE_BUFFER:
            self.preprocessor = preprocessor
            self.device = device
            self.audio_bufferer = BatchedAudioBufferer(
                sample_rate=sample_rate, buffer_size_in_secs=buffer_size_in_secs
            )
            self.right_pad_features = right_pad_features
            self.tail_padding_in_samples = tail_padding_in_samples

    def set_audio_filepaths(self, audio_filepaths: list[str], options: list[RequestOptions]) -> None:
        """
        Set the audio filepaths
        Args:
            audio_filepaths (list[str]): The list of audio filepaths
            options (list[RequestOptions]): The list of options
        """
        self.multi_streamer.set_audio_filepaths(audio_filepaths, options)

    def set_progress_bar(self, progress_bar: ProgressBar) -> None:
        """
        Set the progress bar
        Args:
            progress_bar (ProgressBar): The progress bar to set
        """
        self.multi_streamer.set_progress_bar(progress_bar)

    def get_audio_filepath(self, stream_id: int) -> str:
        """
        Get the audio filepath for a given stream id
        Args:
            stream_id (int): The id of the stream
        Returns:
            str: The audio filepath for the given stream id
        """
        return self.multi_streamer.sid2filepath[stream_id]

    def to_feature_buffers(self, frames: list[Frame]) -> list[FeatureBuffer]:
        """
        Convert frames to feature buffers
        Args:
            frames (list[Frame]): The list of frames
        Returns:
            list[FeatureBuffer]: The list of feature buffers
        """

        # Buffer input frames
        buffered_frames, left_paddings = self.audio_bufferer.update(frames)
        buffers = []

        # If right padding is enabled, convert left paddings to tensor
        if self.right_pad_features:
            left_paddings = torch.tensor(left_paddings, dtype=torch.int64, device=self.device)

        # If right padding is enabled, roll the frames to the left
        for i in range(len(buffered_frames)):
            if self.right_pad_features:
                lpad = left_paddings[i].item()
                if lpad > 0:
                    buffered_frames[i] = buffered_frames[i].roll(shifts=-lpad)
            buffers.append(buffered_frames[i].unsqueeze_(0))

        buffer_lens = torch.tensor([buffers[0].size(1)] * len(buffers), device=self.device)

        # Calculate right paddings and subtract from buffer lens
        # tail_padding_in_samples is used to keep some amount of padding at the end of the buffer
        # some models perform better with this padding
        right_paddings = torch.tensor(
            [frame.size - frame.valid_size - self.tail_padding_in_samples for frame in frames], device=self.device
        ).clamp(min=0)

        # Subtract right paddings from buffer lens
        buffer_lens = buffer_lens - right_paddings

        # If right padding is enabled, subtract left paddings from buffer lens
        # Becouse we rolled the frames to the left
        if self.right_pad_features:
            buffer_lens = buffer_lens - left_paddings

        # Apply preprocessor to get mel spectrograms
        feature_buffers, feature_buffer_lens = self.preprocessor(
            input_signal=torch.cat(buffers).to(self.device), length=buffer_lens
        )

        # Adjust left paddings after preprocessor
        if self.right_pad_features:
            left_paddings = left_paddings / self.preprocessor.featurizer.hop_length
            left_paddings = left_paddings.to(torch.int64)

        return [
            FeatureBuffer(
                features=feature_buffers[i],
                is_first=frame.is_first,
                is_last=frame.is_last,
                stream_id=frame.stream_id,
                right_pad_features=self.right_pad_features,
                length=feature_buffer_lens[i].item(),
                left_padding_length=left_paddings[i].item() if self.right_pad_features else 0,
                options=frame.options,
            )
            for i, frame in enumerate(frames)
        ]

    def __iter__(self) -> Iterator:
        """Returns the iterator object"""
        return self

    def __next__(self) -> list[Request]:
        """Get the next batch of requests.
        Returns:
            list of frames or feature buffers.
        """
        if self.request_type is RequestType.FRAME:
            return next(self.multi_streamer)
        return self.to_feature_buffers(next(self.multi_streamer))
