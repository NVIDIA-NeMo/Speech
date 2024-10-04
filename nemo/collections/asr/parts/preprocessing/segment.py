# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter

import math
import os
import random
from typing import Iterable, List, Optional, Union

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

from nemo.utils import logging

# TODO @blisc: Perhaps refactor instead of import guarding
HAVE_PYDUB = True
try:
    from pydub import AudioSegment as Audio
    from pydub.exceptions import CouldntDecodeError

    # FFMPEG for some formats needs explicitly defined coding-decoding strategy
    ffmpeg_codecs = {'opus': 'opus'}

except ModuleNotFoundError:
    HAVE_PYDUB = False


available_formats = sf.available_formats()
sf_supported_formats = ["." + i.lower() for i in available_formats.keys()]


ChannelSelectorType = Union[int, Iterable[int], str]


def select_channels(signal: npt.NDArray, channel_selector: Optional[ChannelSelectorType] = None) -> npt.NDArray:
    """
    Convert a multi-channel signal to a single-channel signal by averaging over channels or selecting a single channel,
    or pass-through multi-channel signal when channel_selector is `None`.

    Args:
        signal: numpy array with shape (..., num_channels)
        channel selector: string denoting the downmix mode, an integer denoting the channel to be selected, or an iterable
                          of integers denoting a subset of channels. Channel selector is using zero-based indexing.
                          If set to `None`, the original signal will be returned. Uses zero-based indexing.

    Returns:
        numpy array
    """
    if signal.ndim == 1:
        # For one-dimensional input, return the input signal.
        if channel_selector not in [None, 0, 'average']:
            raise ValueError(
                'Input signal is one-dimensional, channel selector (%s) cannot not be used.', str(channel_selector)
            )
        return signal

    num_channels = signal.shape[-1]
    num_samples = signal.size // num_channels  # handle multi-dimensional signals

    if num_channels >= num_samples:
        logging.warning(
            'Number of channels (%d) is greater or equal than number of samples (%d). Check for possible transposition.',
            num_channels,
            num_samples,
        )

    # Samples are arranged as (num_channels, ...)
    if channel_selector is None:
        # keep the original multi-channel signal
        pass
    elif channel_selector == 'average':
        # default behavior: downmix by averaging across channels
        signal = np.mean(signal, axis=-1)
    elif isinstance(channel_selector, int):
        # select a single channel
        if channel_selector >= num_channels:
            raise ValueError(f'Cannot select channel {channel_selector} from a signal with {num_channels} channels.')
        signal = signal[..., channel_selector]
    elif isinstance(channel_selector, Iterable):
        # select multiple channels
        if max(channel_selector) >= num_channels:
            raise ValueError(
                f'Cannot select channel subset {channel_selector} from a signal with {num_channels} channels.'
            )
        signal = signal[..., channel_selector]
        # squeeze the channel dimension if a single-channel is selected
        # this is done to have the same shape as when using integer indexing
        if len(channel_selector) == 1:
            signal = np.squeeze(signal, axis=-1)
    else:
        raise ValueError(f'Unexpected value for channel_selector ({channel_selector})')

    return signal


def get_samples(audio_file: str, target_sr: int = 16000, dtype: str = 'float32'):
    """
    Read the samples from the given audio_file path. If not specified, the input audio file is automatically
    resampled to 16kHz.

    Args:
        audio_file (str):
            Path to the input audio file
        target_sr (int):
            Targeted sampling rate
    Returns:
        samples (numpy.ndarray):
            Time-series sample data from the given audio file
    """
    with sf.SoundFile(audio_file, 'r') as f:
        samples = f.read(dtype=dtype)
        if f.samplerate != target_sr:
            samples = librosa.core.resample(samples, orig_sr=f.samplerate, target_sr=target_sr)
        samples = samples.transpose()
    return samples


class AudioSegment(object):
    """Audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(
        self,
        samples,
        sample_rate,
        target_sr=None,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db: Optional[float] = None,
        ref_channel: Optional[int] = None,
        audio_file: Optional[Union[str, List[str]]] = None,
        offset: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)

        # Check if channel selector is necessary
        if samples.ndim == 1 and channel_selector not in [None, 0, 'average']:
            raise ValueError(
                'Input signal is one-dimensional, channel selector (%s) cannot not be used.', str(channel_selector)
            )
        elif samples.ndim == 2:
            samples = select_channels(samples, channel_selector)
        elif samples.ndim >= 3:
            raise NotImplementedError(
                'Signals with more than two dimensions (sample, channel) are currently not supported.'
            )

        if target_sr is not None and target_sr != sample_rate:
            # resample along the temporal dimension (axis=0) will be in librosa 0.10.0 (#1561)
            samples = samples.transpose()
            samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
            samples = samples.transpose()
            sample_rate = target_sr
        if trim:
            # librosa is using channels-first layout (num_channels, num_samples), which is transpose of AudioSegment's layout
            samples = samples.transpose()
            samples, _ = librosa.effects.trim(
                samples, top_db=trim_top_db, ref=trim_ref, frame_length=trim_frame_length, hop_length=trim_hop_length
            )
            samples = samples.transpose()
        self._samples = samples
        self._sample_rate = sample_rate
        self._orig_sr = orig_sr if orig_sr is not None else sample_rate
        self._ref_channel = ref_channel
        self._normalize_db = normalize_db
        self._audio_file = audio_file
        self._offset = offset
        self._duration = duration
        if normalize_db is not None:
            self.normalize_db(normalize_db, ref_channel)

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        if self.num_channels == 1:
            return "%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB" % (
                type(self),
                self.num_samples,
                self.sample_rate,
                self.duration,
                self.rms_db,
            )
        else:
            rms_db_str = ', '.join([f'{rms:.2f}dB' for rms in self.rms_db])
            return "%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, num_channels=%d, rms=[%s]" % (
                type(self),
                self.num_samples,
                self.sample_rate,
                self.duration,
                self.num_channels,
                rms_db_str,
            )

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(
        cls,
        audio_file,
        target_sr=None,
        int_values=False,
        offset=0,
        duration=0,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db=None,
        ref_channel=None,
    ):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param audio_file: path of file to load.
                           Alternatively, a list of paths of single-channel files can be provided
                           to form a multichannel signal.
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :param trim: if true, trim leading and trailing silence from an audio signal
        :param trim_ref: the reference amplitude. By default, it uses `np.max` and compares to the peak amplitude in
                         the signal
        :param trim_top_db: the threshold (in decibels) below reference to consider as silence
        :param trim_frame_length: the number of samples per analysis frame
        :param trim_hop_length: the number of samples between analysis frames
        :param orig_sr: the original sample rate
        :param channel selector: string denoting the downmix mode, an integer denoting the channel to be selected, or an iterable
                                 of integers denoting a subset of channels. Channel selector is using zero-based indexing.
                                 If set to `None`, the original signal will be used.
        :param normalize_db (Optional[float]): if not None, normalize the audio signal to a target RMS value
        :param ref_channel (Optional[int]): channel to use as reference for normalizing multi-channel audio, set None to use max RMS across channels
        :return: AudioSegment instance
        """
        samples = None
        if isinstance(audio_file, list):
            return cls.from_file_list(
                audio_file_list=audio_file,
                target_sr=target_sr,
                int_values=int_values,
                offset=offset,
                duration=duration,
                trim=trim,
                trim_ref=trim_ref,
                trim_top_db=trim_top_db,
                trim_frame_length=trim_frame_length,
                trim_hop_length=trim_hop_length,
                orig_sr=orig_sr,
                channel_selector=channel_selector,
                normalize_db=normalize_db,
                ref_channel=ref_channel,
            )

        if not isinstance(audio_file, str) or os.path.splitext(audio_file)[-1] in sf_supported_formats:
            try:
                with sf.SoundFile(audio_file, 'r') as f:
                    dtype = 'int32' if int_values else 'float32'
                    sample_rate = f.samplerate
                    if offset is not None and offset > 0:
                        f.seek(int(offset * sample_rate))
                    if duration is not None and duration > 0:
                        samples = f.read(int(duration * sample_rate), dtype=dtype)
                    else:
                        samples = f.read(dtype=dtype)
            except RuntimeError as e:
                logging.error(
                    f"Loading {audio_file} via SoundFile raised RuntimeError: `{e}`. "
                    f"NeMo will fallback to loading via pydub."
                )

                if hasattr(audio_file, "seek"):
                    audio_file.seek(0)

        if HAVE_PYDUB and samples is None:
            try:
                samples = Audio.from_file(audio_file, codec=ffmpeg_codecs.get(os.path.splitext(audio_file)[-1]))
                sample_rate = samples.frame_rate
                num_channels = samples.channels
                if offset is not None and offset > 0:
                    # pydub does things in milliseconds
                    seconds = offset * 1000
                    samples = samples[int(seconds) :]
                if duration is not None and duration > 0:
                    seconds = duration * 1000
                    samples = samples[: int(seconds)]
                samples = np.array(samples.get_array_of_samples())
                # For multi-channel signals, channels are stacked in a one-dimensional vector
                if num_channels > 1:
                    samples = np.reshape(samples, (-1, num_channels))
            except CouldntDecodeError as err:
                logging.error(f"Loading {audio_file} via pydub raised CouldntDecodeError: `{err}`.")

        if samples is None:
            libs = "soundfile, and pydub" if HAVE_PYDUB else "soundfile"
            raise Exception(f"Your audio file {audio_file} could not be decoded. We tried using {libs}.")

        return cls(
            samples,
            sample_rate,
            target_sr=target_sr,
            trim=trim,
            trim_ref=trim_ref,
            trim_top_db=trim_top_db,
            trim_frame_length=trim_frame_length,
            trim_hop_length=trim_hop_length,
            orig_sr=orig_sr,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
            ref_channel=ref_channel,
            audio_file=audio_file,
            offset=offset,
            duration=duration,
        )

    @classmethod
    def from_file_list(
        cls,
        audio_file_list,
        target_sr=None,
        int_values=False,
        offset=0,
        duration=0,
        trim=False,
        channel_selector=None,
        *args,
        **kwargs,
    ):
        """
        Function wrapper for `from_file` method. Load a list of files from `audio_file_list`.
        The length of each audio file is unified with the duration item in the input manifest file.
        See `from_file` method for arguments.

        If a list of files is provided, load samples from individual single-channel files and
        concatenate them along the channel dimension.
        """
        if isinstance(channel_selector, int):
            # Shortcut when selecting a single channel
            if channel_selector >= len(audio_file_list):
                raise RuntimeError(
                    f'Channel cannot be selected: channel_selector={channel_selector}, num_audio_files={len(audio_file_list)}'
                )
            # Select only a single file
            audio_file_list = [audio_file_list[channel_selector]]
            # Reset the channel selector since we applied it here
            channel_selector = None

        samples = None

        for a_file in audio_file_list:
            # Load audio from the current file
            a_segment = cls.from_file(
                audio_file=a_file,
                target_sr=target_sr,
                int_values=int_values,
                offset=offset,
                duration=duration,
                channel_selector=None,
                trim=False,  # Do not apply trim to individual files, it will be applied to the concatenated signal
                *args,
                **kwargs,
            )

            # Only single-channel individual files are supported for now
            if a_segment.num_channels != 1:
                raise RuntimeError(
                    f'Expecting a single-channel audio signal, but loaded {a_segment.num_channels} channels from file {a_file}'
                )

            if target_sr is None:
                # All files need to be loaded with the same sample rate
                target_sr = a_segment.sample_rate

            # Concatenate samples
            a_samples = a_segment.samples[:, None]

            if samples is None:
                samples = a_samples
            else:
                # Check the dimensions match
                if len(a_samples) != len(samples):
                    raise RuntimeError(
                        f'Loaded samples need to have identical length: {a_samples.shape} != {samples.shape}'
                    )

                # Concatenate along channel dimension
                samples = np.concatenate([samples, a_samples], axis=1)

        # Final setup for class initialization
        samples = np.squeeze(samples)
        sample_rate = target_sr

        return cls(
            samples,
            sample_rate,
            target_sr=target_sr,
            trim=trim,
            channel_selector=channel_selector,
            audio_file=audio_file_list,
            *args,
            **kwargs,
        )

    @classmethod
    def segment_from_file(
        cls,
        audio_file,
        target_sr=None,
        n_segments=0,
        trim=False,
        orig_sr=None,
        channel_selector=None,
        offset=None,
        dtype='float32',
    ):
        """Grabs n_segments number of samples from audio_file.
        If offset is not provided, n_segments are selected randomly.
        If offset is provided, it is used to calculate the starting sample.

        Note that audio_file can be either the file path, or a file-like object.

        :param audio_file: path to a file or a file-like object
        :param target_sr: sample rate for the output samples
        :param n_segments: desired number of samples
        :param trim: if true, trim leading and trailing silence from an audio signal
        :param orig_sr: the original sample rate
        :param channel selector: select a subset of channels. If set to `None`, the original signal will be used.
        :param offset: fixed offset in seconds
        :param dtype: data type to load audio as.
        :return: numpy array of samples
        """
        is_segmented = False
        try:
            with sf.SoundFile(audio_file, 'r') as f:
                sample_rate = f.samplerate
                if target_sr is not None:
                    n_segments_at_original_sr = math.ceil(n_segments * sample_rate / target_sr)
                else:
                    n_segments_at_original_sr = n_segments

                if 0 < n_segments_at_original_sr < len(f):
                    max_audio_start = len(f) - n_segments_at_original_sr
                    if offset is None:
                        audio_start = random.randint(0, max_audio_start)
                    else:
                        audio_start = math.floor(offset * sample_rate)
                        if audio_start > max_audio_start:
                            raise RuntimeError(
                                f'Provided audio start ({audio_start}) is larger than the maximum possible ({max_audio_start})'
                            )
                    f.seek(audio_start)
                    samples = f.read(n_segments_at_original_sr, dtype=dtype)
                    is_segmented = True
                elif n_segments_at_original_sr > len(f):
                    logging.warning(
                        f"Number of segments ({n_segments_at_original_sr}) is greater than the length ({len(f)}) of the audio file {audio_file}. This may lead to shape mismatch errors."
                    )
                    samples = f.read(dtype=dtype)
                else:
                    samples = f.read(dtype=dtype)
        except RuntimeError as e:
            logging.error(f"Loading {audio_file} via SoundFile raised RuntimeError: `{e}`.")
            raise e

        features = cls(
            samples, sample_rate, target_sr=target_sr, trim=trim, orig_sr=orig_sr, channel_selector=channel_selector
        )

        if is_segmented:
            features._samples = features._samples[:n_segments]

        return features

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_channels(self):
        if self._samples.ndim == 1:
            return 1
        else:
            return self._samples.shape[-1]

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self.num_samples / float(self._sample_rate)

    @property
    def rms_db(self):
        """Return per-channel RMS value."""
        mean_square = np.mean(self._samples**2, axis=0)
        return 10 * np.log10(mean_square)

    @property
    def orig_sr(self):
        return self._orig_sr

    @property
    def offset(self):
        return float(self._offset) if self._offset is not None else None

    @property
    def audio_file(self):
        return str(self._audio_file) if self._audio_file is not None else None

    def is_empty(self):
        mean_square = np.sum(np.mean(self._samples**2, axis=0))
        return self.num_samples == 0 or mean_square == 0

    def gain_db(self, gain):
        self._samples *= 10.0 ** (gain / 20.0)

    def normalize_db(self, target_db=-20, ref_channel=None):
        """Normalize the signal to a target RMS value in decibels.
        For multi-channel audio, the RMS value is determined by the reference channel (if not None),
        otherwise it will be the maximum RMS across all channels.
        """
        rms_db = self.rms_db
        if self.num_channels > 1:
            rms_db = max(rms_db) if ref_channel is None else rms_db[ref_channel]
        gain = target_db - rms_db
        self.gain_db(gain)

    def pad(self, pad_size, symmetric=False):
        """Add zero padding to the sample. The pad size is given in number
        of samples.
        If symmetric=True, `pad_size` will be added to both sides. If false,
        `pad_size`
        zeros will be added only to the end.
        """
        samples_ndim = self._samples.ndim
        if samples_ndim == 1:
            pad_width = pad_size if symmetric else (0, pad_size)
        elif samples_ndim == 2:
            # pad samples, keep channels
            pad_width = ((pad_size, pad_size), (0, 0)) if symmetric else ((0, pad_size), (0, 0))
        else:
            raise NotImplementedError(
                f"Padding not implemented for signals with more that 2 dimensions. Current samples dimension: {samples_ndim}."
            )
        # apply padding
        self._samples = np.pad(
            self._samples,
            pad_width,
            mode='constant',
        )

    def subsegment(self, start_time=None, end_time=None):
        """Cut the AudioSegment between given boundaries.
        Note that this is an in-place transformation.
        :param start_time: Beginning of subsegment in seconds.
        :type start_time: float
        :param end_time: End of subsegment in seconds.
        :type end_time: float
        :raise ValueError: If start_time or end_time is incorrectly set,
        e.g. out of bounds in time.
        """
        start_time = 0.0 if start_time is None else start_time
        end_time = self.duration if end_time is None else end_time
        if start_time < 0.0:
            start_time = self.duration + start_time
        if end_time < 0.0:
            end_time = self.duration + end_time
        if start_time < 0.0:
            raise ValueError("The slice start position (%f s) is out of bounds." % start_time)
        if end_time < 0.0:
            raise ValueError("The slice end position (%f s) is out of bounds." % end_time)
        if start_time > end_time:
            raise ValueError(
                "The slice start position (%f s) is later than the end position (%f s)." % (start_time, end_time)
            )
        if end_time > self.duration:
            raise ValueError("The slice end position (%f s) is out of bounds (> %f s)" % (end_time, self.duration))
        start_sample = int(round(start_time * self._sample_rate))
        end_sample = int(round(end_time * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]
