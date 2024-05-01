# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import abc
import math
import random
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import librosa
import numpy as np
import torch

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common.parts.preprocessing import collections
from nemo.collections.common.parts.utils import flatten
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, EncodedRepresentation, LengthsType, NeuralType
from nemo.utils import logging

__all__ = [
    'AudioToTargetDataset',
    'AudioToTargetWithReferenceDataset',
    'AudioToTargetWithEmbeddingDataset',
]


def _audio_collate_fn(batch: List[dict]) -> Tuple[torch.Tensor]:
    """Collate a batch of items returned by __getitem__.
    Examples for each signal are zero padded to the same length
    (batch_length), which is determined by the longest example.
    Lengths of the original signals are returned in the output.

    Args:
        batch: List of dictionaries. Each element of the list
            has the following format
            ```
            {
                'signal_0': 1D or 2D tensor,
                'signal_1': 1D or 2D tensor,
                ...
                'signal_N': 1D or 2D tensor,
            }
            ```
            1D tensors have shape (num_samples,) and 2D tensors
            have shape (num_channels, num_samples)

    Returns:
        A tuple containing signal tensor and signal length tensor (in samples)
        for each signal.
        The output has the following format:
        ```
        (signal_0, signal_0_length, signal_1, signal_1_length, ..., signal_N, signal_N_length)
        ```
        Note that the output format is obtained by interleaving signals and their length.
    """
    signals = batch[0].keys()

    batched = tuple()

    for signal in signals:
        signal_length = [b[signal].shape[-1] for b in batch]
        # Batch length is determined by the longest signal in the batch
        batch_length = max(signal_length)
        b_signal = []
        for s_len, b in zip(signal_length, batch):
            # check if padding is necessary
            if s_len < batch_length:
                if b[signal].ndim == 1:
                    # single-channel signal
                    pad = (0, batch_length - s_len)
                elif b[signal].ndim == 2:
                    # multi-channel signal
                    pad = (0, batch_length - s_len, 0, 0)
                else:
                    raise RuntimeError(
                        f'Signal {signal} has unsuported dimensions {signal.shape}. Currently, only 1D and 2D arrays are supported.'
                    )
                b[signal] = torch.nn.functional.pad(b[signal], pad)
            # append the current padded signal
            b_signal.append(b[signal])
        # (signal_batched, signal_length)
        batched += (torch.stack(b_signal), torch.tensor(signal_length, dtype=torch.int32))

    # Currently, outputs are expected to be in a tuple, where each element must correspond
    # to the output type in the OrderedDict returned by output_types.
    #
    # Therefore, we return batched signals by interleaving signals and their length:
    #   (signal_0, signal_0_length, signal_1, signal_1_length, ...)
    return batched


@dataclass
class SignalSetup:
    signals: List[str]  # signal names
    duration: Optional[Union[float, list]] = None  # duration for each signal
    channel_selectors: Optional[List[ChannelSelectorType]] = None  # channel selector for loading each signal


class ASRAudioProcessor:
    """Class that processes an example from Audio collection and returns
    a dictionary with prepared signals.

    For example, the output dictionary may be the following
    ```
    {
        'input_signal': input_signal_tensor,
        'target_signal': target_signal_tensor,
        'reference_signal': reference_signal_tensor,
        'embedding_vector': embedding_vector
    }
    ```
    Keys in the output dictionary are ordered with synchronous signals given first,
    followed by asynchronous signals and embedding.

    Args:
        sample_rate: sample rate used for all audio signals
        random_offset: If `True`, offset will be randomized when loading a subsegment
                       from a file.
        normalization_signal: Normalize all audio with a factor that ensures the signal
                    `example[normalization_signal]` in `process` is in range [-1, 1].
                    All other audio signals are scaled by the same factor. Default is
                    `None`, corresponding to no normalization.
    """

    def __init__(
        self, sample_rate: float, random_offset: bool, normalization_signal: Optional[str] = None, eps: float = 1e-8,
    ):
        self.sample_rate = sample_rate
        self.random_offset = random_offset
        self.normalization_signal = normalization_signal
        self.eps = eps

        self.sync_setup = None
        self.async_setup = None
        self.embedding_setup = None

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float):
        if value <= 0:
            raise ValueError(f'Sample rate must be positive, received {value}')

        self._sample_rate = value

    @property
    def random_offset(self) -> bool:
        return self._random_offset

    @random_offset.setter
    def random_offset(self, value: bool):
        self._random_offset = value

    @property
    def sync_setup(self) -> SignalSetup:
        """Return the current setup for synchronous signals.

        Returns:
            A dataclass containing the list of signals, their
            duration and channel selectors.
        """
        return self._sync_setup

    @sync_setup.setter
    def sync_setup(self, value: Optional[SignalSetup]):
        """Setup signals to be loaded synchronously.

        Args:
            value: An instance of SignalSetup with the following fields
                - signals: list of signals (keys of example.audio_signals) which will be loaded
                           synchronously with the same start time and duration.
                - duration: Duration for each signal to be loaded.
                            If duration is set to None, the whole file will be loaded.
                - channel_selectors: A list of channel selector for each signal. If channel selector
                                     is None, all channels in the audio file will be loaded.
        """
        if value is None or isinstance(value, SignalSetup):
            self._sync_setup = value
        else:
            raise ValueError(f'Unexpected type {type(value)} for value {value}.')

    @property
    def async_setup(self) -> SignalSetup:
        """Return the current setup for asynchronous signals.

        Returns:
            A dataclass containing the list of signals, their
            duration and channel selectors.
        """
        return self._async_setup

    @async_setup.setter
    def async_setup(self, value: Optional[SignalSetup]):
        """Setup signals to be loaded asynchronously.

        Args:
        Args:
            value: An instance of SignalSetup with the following fields
                - signals: list of signals (keys of example.audio_signals) which will be loaded
                           asynchronously with signals possibly having different start and duration
                - duration: Duration for each signal to be loaded.
                            If duration is set to None, the whole file will be loaded.
                - channel_selectors: A list of channel selector for each signal. If channel selector
                                     is None, all channels in the audio file will be loaded.
        """
        if value is None or isinstance(value, SignalSetup):
            self._async_setup = value
        else:
            raise ValueError(f'Unexpected type {type(value)} for value {value}.')

    @property
    def embedding_setup(self) -> SignalSetup:
        """Setup signals corresponding to an embedding vector.
        """
        return self._embedding_setup

    @embedding_setup.setter
    def embedding_setup(self, value: SignalSetup):
        """Setup signals corresponding to an embedding vector.

        Args:
            value: An instance of SignalSetup with the following fields
                - signals: list of signals (keys of example.audio_signals) which will be loaded
                           as embedding vectors.
        """
        if value is None or isinstance(value, SignalSetup):
            self._embedding_setup = value
        else:
            raise ValueError(f'Unexpected type {type(value)} for value {value}.')

    def process(self, example: collections.Audio.OUTPUT_TYPE) -> Dict[str, torch.Tensor]:
        """Process an example from a collection of audio examples.

        Args:
            example: an example from Audio collection.

        Returns:
            An ordered dictionary of signals and their tensors.
            For example, the output dictionary may be the following
            ```
            {
                'input_signal': input_signal_tensor,
                'target_signal': target_signal_tensor,
                'reference_signal': reference_signal_tensor,
                'embedding_vector': embedding_vector
            }
            ```
            Keys in the output dictionary are ordered with synchronous signals given first,
            followed by asynchronous signals and embedding.
        """
        audio = self.load_audio(example=example)
        audio = self.process_audio(audio=audio)
        return audio

    def load_audio(self, example: collections.Audio.OUTPUT_TYPE) -> Dict[str, torch.Tensor]:
        """Given an example, load audio from `example.audio_files` and prepare
        the output dictionary.

        Args:
            example: An example from an audio collection

        Returns:
            An ordered dictionary of signals and their tensors.
            For example, the output dictionary may be the following
            ```
            {
                'input_signal': input_signal_tensor,
                'target_signal': target_signal_tensor,
                'reference_signal': reference_signal_tensor,
                'embedding_vector': embedding_vector
            }
            ```
            Keys in the output dictionary are ordered with synchronous signals given first,
            followed by asynchronous signals and embedding.
        """
        output = OrderedDict()

        if self.sync_setup is not None:
            # Load all signals with the same start and duration
            sync_signals = self.load_sync_signals(example)
            output.update(sync_signals)

        if self.async_setup is not None:
            # Load each signal independently
            async_signals = self.load_async_signals(example)
            output.update(async_signals)

        # Load embedding vector
        if self.embedding_setup is not None:
            embedding = self.load_embedding(example)
            output.update(embedding)

        if not output:
            raise RuntimeError('Output dictionary is empty. Please use `_setup` methods to setup signals to be loaded')

        return output

    def process_audio(self, audio: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process audio signals available in the input dictionary.

        Args:
            audio: A dictionary containing loaded signals `signal: tensor`

        Returns:
            An ordered dictionary of signals and their tensors.
        """
        if self.normalization_signal:
            # Normalize all audio with a factor that ensures the normalization signal is in range [-1, 1].
            norm_scale = audio[self.normalization_signal].abs().max()

            # Do not normalize embeddings
            skip_signals = self.embedding_setup.signals if self.embedding_setup is not None else []

            # Normalize audio signals
            for signal in audio:
                if signal not in skip_signals:
                    # All audio signals are scaled by the same factor.
                    # This ensures that the relative level between signals is preserved.
                    audio[signal] = audio[signal] / (norm_scale + self.eps)

        return audio

    def load_sync_signals(self, example: collections.Audio.OUTPUT_TYPE) -> Dict[str, torch.Tensor]:
        """Load signals with the same start and duration.

        Args:
            example: an example from audio collection

        Returns:
            An ordered dictionary of signals and their tensors.
        """
        output = OrderedDict()
        sync_audio_files = [example.audio_files[s] for s in self.sync_setup.signals]

        sync_samples = self.get_samples_synchronized(
            audio_files=sync_audio_files,
            channel_selectors=self.sync_setup.channel_selectors,
            sample_rate=self.sample_rate,
            duration=self.sync_setup.duration,
            fixed_offset=example.offset,
            random_offset=self.random_offset,
        )

        for signal, samples in zip(self.sync_setup.signals, sync_samples):
            output[signal] = torch.tensor(samples)

        return output

    def load_async_signals(self, example: collections.Audio.OUTPUT_TYPE) -> Dict[str, torch.Tensor]:
        """Load each async signal independently, no constraints on starting
        from the same time.

        Args:
            example: an example from audio collection

        Returns:
            An ordered dictionary of signals and their tensors.
        """
        output = OrderedDict()
        for idx, signal in enumerate(self.async_setup.signals):
            samples = self.get_samples(
                audio_file=example.audio_files[signal],
                sample_rate=self.sample_rate,
                duration=self.async_setup.duration[idx],
                channel_selector=self.async_setup.channel_selectors[idx],
                fixed_offset=example.offset,
                random_offset=self.random_offset,
            )
            output[signal] = torch.tensor(samples)
        return output

    @classmethod
    def get_samples(
        cls,
        audio_file: str,
        sample_rate: int,
        duration: Optional[float] = None,
        channel_selector: ChannelSelectorType = None,
        fixed_offset: float = 0,
        random_offset: bool = False,
    ) -> np.ndarray:
        """Get samples from an audio file.
        For a single-channel signal, the output is shape (num_samples,).
        For a multi-channel signal, the output is shape (num_samples, num_channels).

        Args:
            audio_file: path to an audio file
            sample_rate: desired sample rate for output samples
            duration: Optional desired duration of output samples.
                    If `None`, the complete file will be loaded.
                    If set, a segment of `duration` seconds will be loaded.
            channel_selector: Optional channel selector, for selecting a subset of channels.
            fixed_offset: Optional fixed offset when loading samples.
            random_offset: If `True`, offset will be randomized when loading a short segment
                        from a file. The value is randomized between fixed_offset and
                        max_offset (set depending on the duration and fixed_offset).

        Returns:
            Numpy array with samples from audio file.
            The array has shape (num_samples,) for a single-channel signal
            or (num_channels, num_samples) for a multi-channel signal.
        """
        output = cls.get_samples_synchronized(
            audio_files=[audio_file],
            sample_rate=sample_rate,
            duration=duration,
            channel_selectors=[channel_selector],
            fixed_offset=fixed_offset,
            random_offset=random_offset,
        )

        return output[0]

    @classmethod
    def get_samples_synchronized(
        cls,
        audio_files: List[str],
        sample_rate: int,
        duration: Optional[float] = None,
        channel_selectors: Optional[List[ChannelSelectorType]] = None,
        fixed_offset: float = 0,
        random_offset: bool = False,
    ) -> List[np.ndarray]:
        """Get samples from multiple files with the same start and end point.

        Args:
            audio_files: list of paths to audio files
            sample_rate: desired sample rate for output samples
            duration: Optional desired duration of output samples.
                    If `None`, the complete files will be loaded.
                    If set, a segment of `duration` seconds will be loaded from
                    all files. Segment is synchronized across files, so that
                    start and end points are the same.
            channel_selectors: Optional channel selector for each signal, for selecting
                            a subset of channels.
            fixed_offset: Optional fixed offset when loading samples.
            random_offset: If `True`, offset will be randomized when loading a short segment
                        from a file. The value is randomized between fixed_offset and
                        max_offset (set depending on the duration and fixed_offset).

        Returns:
            List with the same size as `audio_files` but containing numpy arrays
            with samples from each audio file.
            Each array has shape (num_samples,) or (num_channels, num_samples), for single-
            or multi-channel signal, respectively.
            For example, if `audio_files = [path/to/file_1.wav, path/to/file_2.wav]`,
            the output will be a list `output = [samples_file_1, samples_file_2]`.
        """
        if channel_selectors is None:
            channel_selectors = [None] * len(audio_files)

        if duration is None:
            # Load complete files starting from a fixed offset
            offset = fixed_offset  # fixed offset
            num_samples = None  # no constrain on the number of samples

        else:
            # Fixed duration of the output
            audio_durations = cls.get_duration(audio_files)
            min_audio_duration = min(audio_durations)
            available_duration = min_audio_duration - fixed_offset

            if available_duration <= 0:
                raise ValueError(f'Fixed offset {fixed_offset}s is larger than shortest file {min_duration}s.')

            if duration + fixed_offset > min_audio_duration:
                # The shortest file is shorter than the requested duration
                logging.debug(
                    f'Shortest file ({min_audio_duration}s) is less than the desired duration {duration}s + fixed offset {fixed_offset}s. Returned signals will be shortened to {available_duration} seconds.'
                )
                offset = fixed_offset
                duration = available_duration
            elif random_offset:
                # Randomize offset based on the shortest file
                max_offset = min_audio_duration - duration
                offset = random.uniform(fixed_offset, max_offset)
            else:
                # Fixed offset
                offset = fixed_offset

            # Fixed number of samples
            num_samples = math.floor(duration * sample_rate)

        output = []

        # Prepare segments
        for idx, audio_file in enumerate(audio_files):
            segment_samples = cls.get_samples_from_file(
                audio_file=audio_file,
                sample_rate=sample_rate,
                offset=offset,
                num_samples=num_samples,
                channel_selector=channel_selectors[idx],
            )
            output.append(segment_samples)

        return output

    @classmethod
    def get_samples_from_file(
        cls,
        audio_file: Union[str, List[str]],
        sample_rate: int,
        offset: float,
        num_samples: Optional[int] = None,
        channel_selector: Optional[ChannelSelectorType] = None,
    ) -> np.ndarray:
        """Get samples from a single or multiple files.
        If loading samples from multiple files, they will
        be concatenated along the channel dimension.

        Args:
            audio_file: path or a list of paths.
            sample_rate: sample rate of the loaded samples
            offset: fixed offset in seconds
            num_samples: Optional, number of samples to load.
                         If `None`, all available samples will be loaded.
            channel_selector: Select a subset of available channels.

        Returns:
            An array with shape (samples,) or (channels, samples)
        """
        if isinstance(audio_file, str):
            # Load samples from a single file
            segment_samples = cls.get_segment_from_file(
                audio_file=audio_file,
                sample_rate=sample_rate,
                offset=offset,
                num_samples=num_samples,
                channel_selector=channel_selector,
            )
        elif isinstance(audio_file, list):
            # Load samples from multiple files and form a multi-channel signal
            segment_samples = []
            for a_file in audio_file:
                a_file_samples = cls.get_segment_from_file(
                    audio_file=a_file,
                    sample_rate=sample_rate,
                    offset=offset,
                    num_samples=num_samples,
                    channel_selector=channel_selector,
                )
                segment_samples.append(a_file_samples)
            segment_samples = cls.list_to_multichannel(segment_samples)
        elif audio_file is None:
            # Support for inference, when the target signal is `None`
            segment_samples = []
        else:
            raise RuntimeError(f'Unexpected audio_file type {type(audio_file)}')
        return segment_samples

    @staticmethod
    def get_segment_from_file(
        audio_file: str,
        sample_rate: int,
        offset: float,
        num_samples: Optional[int] = None,
        channel_selector: Optional[ChannelSelectorType] = None,
    ) -> np.ndarray:
        """Get a segment of samples from a single audio file.

        Args:
            audio_file: path to an audio file
            sample_rate: sample rate of the loaded samples
            offset: fixed offset in seconds
            num_samples: Optional, number of samples to load.
                         If `None`, all available samples will be loaded.
            channel_selector: Select a subset of available channels.

        Returns:
           An array with shape (samples,) or (channels, samples) 
        """
        if num_samples is None:
            segment = AudioSegment.from_file(
                audio_file=audio_file, target_sr=sample_rate, offset=offset, channel_selector=channel_selector,
            )

        else:
            segment = AudioSegment.segment_from_file(
                audio_file=audio_file,
                target_sr=sample_rate,
                n_segments=num_samples,
                offset=offset,
                channel_selector=channel_selector,
            )

        if segment.samples.ndim == 1:
            # Single-channel signal
            return segment.samples
        elif segment.samples.ndim == 2:
            # Use multi-channel format as (channels, samples)
            return segment.samples.T
        else:
            raise RuntimeError(f'Unexpected samples shape: {segment.samples.shape}')

    @staticmethod
    def list_to_multichannel(signal: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Convert a list of signals into a multi-channel signal by concatenating
        the elements of the list along the channel dimension.

        If input is not a list, it is returned unmodified.

        Args:
            signal: list of arrays

        Returns:
            Numpy array obtained by concatenating the elements of the list
            along the channel dimension (axis=0).
        """
        if not isinstance(signal, list):
            # Nothing to do there
            return signal
        elif len(signal) == 0:
            # Nothing to do, return as is
            return signal
        elif len(signal) == 1:
            # Nothing to concatenate, return the original format
            return signal[0]

        # If multiple signals are provided in a list, we concatenate them along the channel dimension
        if signal[0].ndim == 1:
            # Single-channel individual files
            mc_signal = np.stack(signal, axis=0)
        elif signal[0].ndim == 2:
            # Multi-channel individual files
            mc_signal = np.concatenate(signal, axis=0)
        else:
            raise RuntimeError(f'Unexpected target with {signal[0].ndim} dimensions.')

        return mc_signal

    @staticmethod
    def get_duration(audio_files: List[str]) -> List[float]:
        """Get duration for each audio file in `audio_files`.

        Args:
            audio_files: list of paths to audio files

        Returns:
            List of durations in seconds.
        """
        duration = [librosa.get_duration(path=f) for f in flatten(audio_files)]
        return duration

    def load_embedding(self, example: collections.Audio.OUTPUT_TYPE) -> Dict[str, torch.Tensor]:
        """Given an example, load embedding from `example.audio_files[embedding]`
        and return it in a dictionary.

        Args:
            example: An example from audio collection

        Returns:
            An dictionary of embedding keys and their tensors.
        """
        output = OrderedDict()
        for idx, signal in enumerate(self.embedding_setup.signals):
            embedding_file = example.audio_files[signal]
            embedding = self.load_embedding_vector(embedding_file)
            output[signal] = torch.tensor(embedding)
        return output

    @staticmethod
    def load_embedding_vector(filepath: str) -> np.ndarray:
        """Load an embedding vector from a file.

        Args:
            filepath: path to a file storing a vector.
                    Currently, it is assumed the file is a npy file.
        
        Returns:
            Array loaded from filepath.
        """
        if filepath.endswith('.npy'):
            with open(filepath, 'rb') as f:
                embedding = np.load(f)
        else:
            raise RuntimeError(f'Unknown embedding file format in file: {filepath}')

        return embedding


class BaseAudioDataset(Dataset):
    """Base class of audio datasets, providing common functionality
    for other audio datasets.

    Args:
        collection: Collection of audio examples prepared from manifest files.
        audio_processor: Used to process every example from the collection.
                         A callable with `process` method. For reference,
                         please check ASRAudioProcessor.
    """

    @property
    @abc.abstractmethod
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """

    def __init__(self, collection: collections.Audio, audio_processor: Callable, output_type: Type[namedtuple]):
        """Instantiates an audio dataset.
        """
        super().__init__()

        self.collection = collection
        self.audio_processor = audio_processor
        self.output_type = output_type

    def num_channels(self, signal_key) -> int:
        """Returns the number of channels for a particular signal in
        items prepared by this dictionary.

        More specifically, this will get the tensor from the first
        item in the dataset, check if it's a one- or two-dimensional
        tensor, and return the number of channels based on the size
        of the first axis (shape[0]).

        NOTE:
        This assumes that all examples have the same number of channels.
        
        Args:
            signal_key: string, used to select a signal from the dictionary
                        output by __getitem__

        Returns:
            Number of channels for the selected signal.
        """
        # Assumption: whole dataset has the same number of channels
        item = self.__getitem__(0)

        if item[signal_key].ndim == 1:
            return 1
        elif item[signal_key].ndim == 2:
            return item[signal_key].shape[0]
        else:
            raise RuntimeError(
                f'Unexpected number of dimension for signal {signal_key} with shape {item[signal_key].shape}'
            )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a single example from the dataset.

        Args:
            index: integer index of an example in the collection

        Returns:
            Dictionary providing mapping from signal to its tensor.
            For example:
            ```
            {
                'input_signal': input_signal_tensor,
                'target_signal': target_signal_tensor,
            }
            ```
        """
        example = self.collection[index]
        output = self.audio_processor.process(example=example)

        return output

    def __len__(self) -> int:
        """Return the number of examples in the dataset.
        """
        return len(self.collection)

    def _collate_fn(self, batch) -> Tuple[torch.Tensor]:
        """Collate items in a batch.
        """
        return self.output_type(*_audio_collate_fn(batch))


AudioToTargetExample = namedtuple(
    typename='AudioToTargetExample', field_names='input_signal input_length target_signal target_length'
)


class AudioToTargetDataset(BaseAudioDataset):
    """A dataset for audio-to-audio tasks where the goal is to use
    an input signal to recover the corresponding target signal.

    Each line of the manifest file is expected to have the following format
        ```
        {
            'input_key': 'path/to/input.wav',
            'target_key': 'path/to/path_to_target.wav',
            'duration': duration_of_input,
        }
        ```

    Additionally, multiple audio files may be provided for each key in the manifest, for example,
        ```
        {
            'input_key': 'path/to/input.wav',
            'target_key': ['path/to/path_to_target_ch0.wav', 'path/to/path_to_target_ch1.wav'],
            'duration': duration_of_input,
        }
        ```

    Keys for input and target signals can be configured in the constructor (`input_key` and `target_key`).

    Args:
        manifest_filepath: Path to manifest file in a format described above.
        sample_rate: Sample rate for loaded audio signals.
        input_key: Key pointing to input audio files in the manifest
        target_key: Key pointing to target audio files in manifest
        audio_duration: Optional duration of each item returned by __getitem__.
                        If `None`, complete audio will be loaded.
                        If set, a random subsegment will be loaded synchronously from
                        target and audio, i.e., with the same start and end point.
        random_offset: If `True`, offset will be randomized when loading a subsegment
                       from a file.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        input_channel_selector: Optional, select subset of channels from each input audio file.
                                If `None`, all channels will be loaded.
        target_channel_selector: Optional, select subset of channels from each input audio file.
                                 If `None`, all channels will be loaded.
        normalization_signal: Normalize audio signals with a scale that ensures the normalization signal is in range [-1, 1].
                              All audio signals are scaled by the same factor. Supported values are `None` (no normalization),
                              'input_signal', 'target_signal'.
    """

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        input_key: str,
        target_key: str,
        audio_duration: Optional[float] = None,
        random_offset: bool = False,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        input_channel_selector: Optional[int] = None,
        target_channel_selector: Optional[int] = None,
        normalization_signal: Optional[str] = None,
    ):
        audio_to_manifest_key = {
            'input_signal': input_key,
            'target_signal': target_key,
        }

        collection = collections.AudioCollection(
            manifest_files=manifest_filepath,
            audio_to_manifest_key=audio_to_manifest_key,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        audio_processor = ASRAudioProcessor(
            sample_rate=sample_rate, random_offset=random_offset, normalization_signal=normalization_signal,
        )
        audio_processor.sync_setup = SignalSetup(
            signals=['input_signal', 'target_signal'],
            duration=audio_duration,
            channel_selectors=[input_channel_selector, target_channel_selector],
        )

        super().__init__(collection=collection, audio_processor=audio_processor, output_type=AudioToTargetExample)

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.

        Returns:
            Ordered dictionary in the following form:
            ```
            {
                'input_signal': batched single- or multi-channel format,
                'input_length': batched original length of each input signal
                'target_signal': batched single- or multi-channel format,
                'target_length': batched original length of each target signal                
            }
            ```
        """
        sc_audio_type = NeuralType(('B', 'T'), AudioSignal())
        mc_audio_type = NeuralType(('B', 'C', 'T'), AudioSignal())

        return OrderedDict(
            input_signal=sc_audio_type if self.num_channels('input_signal') == 1 else mc_audio_type,
            input_length=NeuralType(('B',), LengthsType()),
            target_signal=sc_audio_type if self.num_channels('target_signal') == 1 else mc_audio_type,
            target_length=NeuralType(('B',), LengthsType()),
        )


AudioToTargetWithReferenceExample = namedtuple(
    typename='AudioToTargetWithReferenceExample',
    field_names='input_signal input_length target_signal target_length reference_signal reference_length',
)


class AudioToTargetWithReferenceDataset(BaseAudioDataset):
    """A dataset for audio-to-audio tasks where the goal is to use
    an input signal to recover the corresponding target signal and an
    additional reference signal is available.

    This can be used, for example, when a reference signal is
    available from
    - enrollment utterance for the target signal
    - echo reference from playback
    - reference from another sensor that correlates with the target signal

    Each line of the manifest file is expected to have the following format
        ```
        {
            'input_key': 'path/to/input.wav',
            'target_key': 'path/to/path_to_target.wav',
            'reference_key': 'path/to/path_to_reference.wav',
            'duration': duration_of_input,
        }
        ```

    Keys for input, target and reference signals can be configured in the constructor.

    Args:
        manifest_filepath: Path to manifest file in a format described above.
        sample_rate: Sample rate for loaded audio signals.
        input_key: Key pointing to input audio files in the manifest
        target_key: Key pointing to target audio files in manifest
        reference_key: Key pointing to reference audio files in manifest
        audio_duration: Optional duration of each item returned by __getitem__.
                        If `None`, complete audio will be loaded.
                        If set, a random subsegment will be loaded synchronously from
                        target and audio, i.e., with the same start and end point.
        random_offset: If `True`, offset will be randomized when loading a subsegment
                       from a file.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        input_channel_selector: Optional, select subset of channels from each input audio file.
                                If `None`, all channels will be loaded.
        target_channel_selector: Optional, select subset of channels from each input audio file.
                                 If `None`, all channels will be loaded.
        reference_channel_selector: Optional, select subset of channels from each input audio file.
                                    If `None`, all channels will be loaded.
        reference_is_synchronized: If True, it is assumed that the reference signal is synchronized
                                   with the input signal, so the same subsegment will be loaded as for
                                   input and target. If False, reference signal will be loaded independently
                                   from input and target.
        reference_duration: Optional, can be used to set a fixed duration of the reference utterance. If `None`,
                            complete audio file will be loaded.
        normalization_signal: Normalize audio signals with a scale that ensures the normalization signal is in range [-1, 1].
                              All audio signals are scaled by the same factor. Supported values are `None` (no normalization),
                              'input_signal', 'target_signal', 'reference_signal'.
    """

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        input_key: str,
        target_key: str,
        reference_key: str,
        audio_duration: Optional[float] = None,
        random_offset: bool = False,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        input_channel_selector: Optional[int] = None,
        target_channel_selector: Optional[int] = None,
        reference_channel_selector: Optional[int] = None,
        reference_is_synchronized: bool = True,
        reference_duration: Optional[float] = None,
        normalization_signal: Optional[str] = None,
    ):
        audio_to_manifest_key = {
            'input_signal': input_key,
            'target_signal': target_key,
            'reference_signal': reference_key,
        }

        collection = collections.AudioCollection(
            manifest_files=manifest_filepath,
            audio_to_manifest_key=audio_to_manifest_key,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        audio_processor = ASRAudioProcessor(
            sample_rate=sample_rate, random_offset=random_offset, normalization_signal=normalization_signal,
        )

        if reference_is_synchronized:
            audio_processor.sync_setup = SignalSetup(
                signals=['input_signal', 'target_signal', 'reference_signal'],
                duration=audio_duration,
                channel_selectors=[input_channel_selector, target_channel_selector, reference_channel_selector],
            )
        else:
            audio_processor.sync_setup = SignalSetup(
                signals=['input_signal', 'target_signal'],
                duration=audio_duration,
                channel_selectors=[input_channel_selector, target_channel_selector],
            )
            audio_processor.async_setup = SignalSetup(
                signals=['reference_signal'],
                duration=[reference_duration],
                channel_selectors=[reference_channel_selector],
            )

        super().__init__(
            collection=collection, audio_processor=audio_processor, output_type=AudioToTargetWithReferenceExample
        )

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.

        Returns:
            Ordered dictionary in the following form:
            ```
            {
                'input_signal': batched single- or multi-channel format,
                'input_length': batched original length of each input signal
                'target_signal': batched single- or multi-channel format,
                'target_length': batched original length of each target signal
                'reference_signal': single- or multi-channel format,
                'reference_length': original length of each reference signal
            }
            ```
        """
        sc_audio_type = NeuralType(('B', 'T'), AudioSignal())
        mc_audio_type = NeuralType(('B', 'C', 'T'), AudioSignal())

        return OrderedDict(
            input_signal=sc_audio_type if self.num_channels('input_signal') == 1 else mc_audio_type,
            input_length=NeuralType(('B',), LengthsType()),
            target_signal=sc_audio_type if self.num_channels('target_signal') == 1 else mc_audio_type,
            target_length=NeuralType(('B',), LengthsType()),
            reference_signal=sc_audio_type if self.num_channels('reference_signal') == 1 else mc_audio_type,
            reference_length=NeuralType(('B',), LengthsType()),
        )


AudioToTargetWithEmbeddingExample = namedtuple(
    typename='AudioToTargetWithEmbeddingExample',
    field_names='input_signal input_length target_signal target_length embedding_vector embedding_length',
)


class AudioToTargetWithEmbeddingDataset(BaseAudioDataset):
    """A dataset for audio-to-audio tasks where the goal is to use
    an input signal to recover the corresponding target signal and an
    additional embedding signal. It is assumed that the embedding
    is in a form of a vector.

    Each line of the manifest file is expected to have the following format
        ```
        {
            input_key: 'path/to/input.wav',
            target_key: 'path/to/path_to_target.wav',
            embedding_key: 'path/to/path_to_reference.npy',
            'duration': duration_of_input,
        }
        ```

    Keys for input, target and embedding signals can be configured in the constructor.

    Args:
        manifest_filepath: Path to manifest file in a format described above.
        sample_rate: Sample rate for loaded audio signals.
        input_key: Key pointing to input audio files in the manifest
        target_key: Key pointing to target audio files in manifest
        embedding_key: Key pointing to embedding files in manifest
        audio_duration: Optional duration of each item returned by __getitem__.
                        If `None`, complete audio will be loaded.
                        If set, a random subsegment will be loaded synchronously from
                        target and audio, i.e., with the same start and end point.
        random_offset: If `True`, offset will be randomized when loading a subsegment
                       from a file.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        input_channel_selector: Optional, select subset of channels from each input audio file.
                                If `None`, all channels will be loaded.
        target_channel_selector: Optional, select subset of channels from each input audio file.
                                 If `None`, all channels will be loaded.
        normalization_signal: Normalize audio signals with a scale that ensures the normalization signal is in range [-1, 1].
                              All audio signals are scaled by the same factor. Supported values are `None` (no normalization),
                              'input_signal', 'target_signal'.
    """

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        input_key: str,
        target_key: str,
        embedding_key: str,
        audio_duration: Optional[float] = None,
        random_offset: bool = False,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        input_channel_selector: Optional[int] = None,
        target_channel_selector: Optional[int] = None,
        normalization_signal: Optional[str] = None,
    ):
        audio_to_manifest_key = {
            'input_signal': input_key,
            'target_signal': target_key,
            'embedding_vector': embedding_key,
        }

        collection = collections.AudioCollection(
            manifest_files=manifest_filepath,
            audio_to_manifest_key=audio_to_manifest_key,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        audio_processor = ASRAudioProcessor(
            sample_rate=sample_rate, random_offset=random_offset, normalization_signal=normalization_signal,
        )
        audio_processor.sync_setup = SignalSetup(
            signals=['input_signal', 'target_signal'],
            duration=audio_duration,
            channel_selectors=[input_channel_selector, target_channel_selector],
        )
        audio_processor.embedding_setup = SignalSetup(signals=['embedding_vector'])

        super().__init__(
            collection=collection, audio_processor=audio_processor, output_type=AudioToTargetWithEmbeddingExample
        )

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.

        Returns:
            Ordered dictionary in the following form:
            ```
            {
                'input_signal': batched single- or multi-channel format,
                'input_length': batched original length of each input signal
                'target_signal': batched single- or multi-channel format,
                'target_length': batched original length of each target signal
                'embedding_vector': batched embedded vector format,
                'embedding_length': batched original length of each embedding vector
            }
            ```
        """
        sc_audio_type = NeuralType(('B', 'T'), AudioSignal())
        mc_audio_type = NeuralType(('B', 'C', 'T'), AudioSignal())

        return OrderedDict(
            input_signal=sc_audio_type if self.num_channels('input_signal') == 1 else mc_audio_type,
            input_length=NeuralType(('B',), LengthsType()),
            target_signal=sc_audio_type if self.num_channels('target_signal') == 1 else mc_audio_type,
            target_length=NeuralType(('B',), LengthsType()),
            embedding_vector=NeuralType(('B', 'D'), EncodedRepresentation()),
            embedding_length=NeuralType(('B',), LengthsType()),
        )
