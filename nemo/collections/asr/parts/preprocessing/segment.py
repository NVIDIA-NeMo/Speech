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

import librosa
import numpy as np
import soundfile as sf

from nemo.collections.asr.parts.utils.audio_utils import select_channels
from nemo.utils import logging

# TODO @blisc: Perhaps refactor instead of import guarding
HAVE_PYDUB = True
try:
    from pydub import AudioSegment as Audio
    from pydub.exceptions import CouldntDecodeError
except ModuleNotFoundError:
    HAVE_PYDUB = False


available_formats = sf.available_formats()
sf_supported_formats = ["." + i.lower() for i in available_formats.keys()]


class AudioSegment(object):
    """Monaural audio segment abstraction.
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
    ):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param audio_file: path of file to load
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
        :return: numpy array of samples
        """
        samples = None
        if not isinstance(audio_file, str) or os.path.splitext(audio_file)[-1] in sf_supported_formats:
            try:
                with sf.SoundFile(audio_file, 'r') as f:
                    dtype = 'int32' if int_values else 'float32'
                    sample_rate = f.samplerate
                    if offset > 0:
                        f.seek(int(offset * sample_rate))
                    if duration > 0:
                        samples = f.read(int(duration * sample_rate), dtype=dtype)
                    else:
                        samples = f.read(dtype=dtype)
            except RuntimeError as e:
                logging.error(
                    f"Loading {audio_file} via SoundFile raised RuntimeError: `{e}`. "
                    f"NeMo will fallback to loading via pydub."
                )

        if HAVE_PYDUB and samples is None:
            try:
                samples = Audio.from_file(audio_file)
                sample_rate = samples.frame_rate
                num_channels = samples.channels
                if offset > 0:
                    # pydub does things in milliseconds
                    seconds = offset * 1000
                    samples = samples[int(seconds) :]
                if duration > 0:
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
        )

    @classmethod
    def segment_from_file(
        cls, audio_file, target_sr=None, n_segments=0, trim=False, orig_sr=None, channel_selector=None, offset=None
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
                                f'Provided audio start ({audio_start_seconds} seconds = {audio_start} samples) is larger than the maximum possible ({max_audio_start})'
                            )
                    f.seek(audio_start)
                    samples = f.read(n_segments_at_original_sr, dtype='float32')
                    is_segmented = True
                elif n_segments_at_original_sr > len(f):
                    logging.warning(
                        f"Number of segments ({n_segments_at_original_sr}) is greater than the length ({len(f)}) of the audio file {audio_file}. This may lead to shape mismatch errors."
                    )
                    samples = f.read(dtype='float32')
                else:
                    samples = f.read(dtype='float32')
        except RuntimeError as e:
            logging.error(f"Loading {audio_file} via SoundFile raised RuntimeError: `{e}`.")

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
        """Return per-channel RMS value.
        """
        mean_square = np.mean(self._samples ** 2, axis=0)
        return 10 * np.log10(mean_square)

    @property
    def orig_sr(self):
        return self._orig_sr

    def gain_db(self, gain):
        self._samples *= 10.0 ** (gain / 20.0)

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
        self._samples = np.pad(self._samples, pad_width, mode='constant',)

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
