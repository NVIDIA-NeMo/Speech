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
import copy
import inspect
import io
import os
import random
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import IterableDataset
from nemo.utils import logging

# TODO @blisc: Perhaps refactor instead of import guarding
HAVE_OMEGACONG_WEBDATASET = True
try:
    import webdataset as wd
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:
    from nemo.utils.exceptions import LightningNotInstalledException

    HAVE_OMEGACONG_WEBDATASET = False


try:
    from nemo.collections.asr.parts.utils import numba_utils

    HAVE_NUMBA = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMBA = False


def read_one_audiosegment(manifest, target_sr, tarred_audio=False, audio_dataset=None):
    if tarred_audio:
        if audio_dataset is None:
            raise TypeError("Expected augmentation dataset but got None")
        audio_file, file_id, manifest_entry = next(audio_dataset)

        offset = 0 if manifest_entry.offset is None else manifest_entry.offset
        duration = 0 if manifest_entry.duration is None else manifest_entry.duration

    else:
        audio_record = random.sample(manifest.data, 1)[0]
        audio_file = audio_record.audio_file
        offset = 0 if audio_record.offset is None else audio_record.offset
        duration = 0 if audio_record.duration is None else audio_record.duration

    return AudioSegment.from_file(audio_file, target_sr=target_sr, offset=offset, duration=duration)


class Perturbation(object):
    def max_augmentation_length(self, length):
        return length

    def perturb(self, data):
        raise NotImplementedError


class SpeedPerturbation(Perturbation):
    """
    Performs Speed Augmentation by re-sampling the data to a different sampling rate,
    which does not preserve pitch.

    Note: This is a very slow operation for online augmentation. If space allows,
    it is preferable to pre-compute and save the files to augment the dataset.

    Args:
        sr: Original sampling rate.
        resample_type: Type of resampling operation that will be performed.
            For better speed using `resampy`'s fast resampling method, use `resample_type='kaiser_fast'`.
            For high-quality resampling, set `resample_type='kaiser_best'`.
            To use `scipy.signal.resample`, set `resample_type='fft'` or `resample_type='scipy'`
        min_speed_rate: Minimum sampling rate modifier.
        max_speed_rate: Maximum sampling rate modifier.
        num_rates: Number of discrete rates to allow. Can be a positive or negative
            integer.
            If a positive integer greater than 0 is provided, the range of
            speed rates will be discretized into `num_rates` values.
            If a negative integer or 0 is provided, the full range of speed rates
            will be sampled uniformly.
            Note: If a positive integer is provided and the resultant discretized
            range of rates contains the value '1.0', then those samples with rate=1.0,
            will not be augmented at all and simply skipped. This is to unnecessary
            augmentation and increase computation time. Effective augmentation chance
            in such a case is = `prob * (num_rates - 1 / num_rates) * 100`% chance
            where `prob` is the global probability of a sample being augmented.
        rng: Random seed. Default is None
    """

    def __init__(self, sr, resample_type, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=5, rng=None):

        min_rate = min(min_speed_rate, max_speed_rate)
        if min_rate < 0.0:
            raise ValueError("Minimum sampling rate modifier must be > 0.")

        if resample_type not in ('kaiser_best', 'kaiser_fast', 'fft', 'scipy'):
            raise ValueError("Supported `resample_type` values are ('kaiser_best', 'kaiser_fast', 'fft', 'scipy')")

        self._sr = sr
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_rate, self._max_rate, self._num_rates, endpoint=True)
        self._res_type = resample_type
        random.seed(rng) if rng else None

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        # Select speed rate either from choice or random sample
        if self._num_rates < 0:
            speed_rate = random.uniform(self._min_rate, self._max_rate)
        else:
            speed_rate = random.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return

        new_sr = int(self._sr * speed_rate)
        data._samples = librosa.core.resample(
            data._samples, orig_sr=self._sr, target_sr=new_sr, res_type=self._res_type
        )


class TimeStretchPerturbation(Perturbation):
    """
    Time-stretch an audio series by a fixed rate while preserving pitch, based on [1, 2].

    Note:
    This is a simplified implementation, intended primarily for reference and pedagogical purposes.
    It makes no attempt to handle transients, and is likely to produce audible artifacts.

    Reference
    [1] [Ellis, D. P. W. “A phase vocoder in Matlab.” Columbia University, 2002.]
    (http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/)
    [2] [librosa.effects.time_stretch]
    (https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html)

    Args:
        min_speed_rate: Minimum sampling rate modifier.
        max_speed_rate: Maximum sampling rate modifier.
        num_rates: Number of discrete rates to allow. Can be a positive or negative
            integer.
            If a positive integer greater than 0 is provided, the range of
            speed rates will be discretized into `num_rates` values.
            If a negative integer or 0 is provided, the full range of speed rates
            will be sampled uniformly.
            Note: If a positive integer is provided and the resultant discretized
            range of rates contains the value '1.0', then those samples with rate=1.0,
            will not be augmented at all and simply skipped. This is to avoid unnecessary
            augmentation and increase computation time. Effective augmentation chance
            in such a case is = `prob * (num_rates - 1 / num_rates) * 100`% chance
            where `prob` is the global probability of a sample being augmented.
        n_fft: Number of fft filters to be computed.
        rng: Random seed. Default is None
    """

    def __init__(self, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=5, n_fft=512, rng=None):

        min_rate = min(min_speed_rate, max_speed_rate)
        if min_rate < 0.0:
            raise ValueError("Minimum sampling rate modifier must be > 0.")

        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_rate, self._max_rate, self._num_rates, endpoint=True)
        random.seed(rng) if rng else None

        # Pre-compute constants
        self._n_fft = int(n_fft)
        self._hop_length = int(n_fft // 2)

        # Pre-allocate buffers
        self._phi_advance_fast = np.linspace(0, np.pi * self._hop_length, self._hop_length + 1)
        self._scale_buffer_fast = np.empty(self._hop_length + 1, dtype=np.float32)

        self._phi_advance_slow = np.linspace(0, np.pi * self._n_fft, self._n_fft + 1)
        self._scale_buffer_slow = np.empty(self._n_fft + 1, dtype=np.float32)

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        # Select speed rate either from choice or random sample
        if self._num_rates < 0:
            speed_rate = random.uniform(self._min_rate, self._max_rate)
        else:
            speed_rate = random.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return

        # Increase `n_fft` based on task (speed up or slow down audio)
        # This greatly reduces upper bound of maximum time taken
        # to compute slowed down audio segments.
        if speed_rate >= 1.0:  # Speed up audio
            fft_multiplier = 1
            phi_advance = self._phi_advance_fast
            scale_buffer = self._scale_buffer_fast

        else:  # Slow down audio
            fft_multiplier = 2
            phi_advance = self._phi_advance_slow
            scale_buffer = self._scale_buffer_slow

        n_fft = int(self._n_fft * fft_multiplier)
        hop_length = int(self._hop_length * fft_multiplier)

        # Perform short-term Fourier transform (STFT)
        stft = librosa.core.stft(data._samples, n_fft=n_fft, hop_length=hop_length)

        # Stretch by phase vocoding
        if HAVE_NUMBA:
            stft_stretch = numba_utils.phase_vocoder(stft, speed_rate, phi_advance, scale_buffer)

        else:
            stft_stretch = librosa.core.phase_vocoder(stft, speed_rate, hop_length)

        # Predict the length of y_stretch
        len_stretch = int(round(len(data._samples) / speed_rate))

        # Invert the STFT
        y_stretch = librosa.core.istft(
            stft_stretch, dtype=data._samples.dtype, hop_length=hop_length, length=len_stretch
        )

        data._samples = y_stretch


class SilencePerturbation(Perturbation):
    """
    Applies random silence at the start and/or end of the audio.

    Args:
        min_start_silence_secs (float): Min start silence level in secs
        max_start_silence_secs (float): Max start silence level in secs
        min_end_silence_secs (float): Min end silence level in secs
        max_end_silence_secs (float): Max end silence level in secs
        rng (int): Random seed. Default is None
        value: (float): value representing silence to be added to audio array.
    """

    def __init__(
        self,
        min_start_silence_secs: float = 0,
        max_start_silence_secs: float = 0,
        min_end_silence_secs: float = 0,
        max_end_silence_secs: float = 0,
        rng: int = None,
        value: float = 0,
    ):
        self._min_start_silence_secs = min_start_silence_secs
        self._max_start_silence_secs = max_start_silence_secs
        self._min_end_silence_secs = min_end_silence_secs
        self._max_end_silence_secs = max_end_silence_secs

        random.seed(rng) if rng else None
        self._value = value

    def perturb(self, data):
        start_silence_len = random.uniform(self._min_start_silence_secs, self._max_start_silence_secs)
        end_silence_len = random.uniform(self._min_end_silence_secs, self._max_end_silence_secs)
        start = np.full((int(start_silence_len * data.sample_rate),), self._value)
        end = np.full((int(end_silence_len * data.sample_rate),), self._value)

        data._samples = np.concatenate([start, data._samples, end])


class GainPerturbation(Perturbation):
    """
    Applies random gain to the audio.

    Args:
        min_gain_dbfs (float): Min gain level in dB
        max_gain_dbfs (float): Max gain level in dB
        rng (int): Random seed. Default is None
    """

    def __init__(self, min_gain_dbfs=-10, max_gain_dbfs=10, rng=None):
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs
        random.seed(rng) if rng else None

    def perturb(self, data):
        gain = random.uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        data._samples = data._samples * (10.0 ** (gain / 20.0))


class ImpulsePerturbation(Perturbation):
    """
    Convolves audio with a Room Impulse Response.

    Args:
        manifest_path (list): Manifest file for RIRs
        audio_tar_filepaths (list): Tar files, if RIR audio files are tarred
        shuffle_n (int): Shuffle parameter for shuffling buffered files from the tar files
        normalize_impulse (bool): Normalize impulse response to zero mean and amplitude 1
        shift_impulse (bool): Shift impulse response to adjust for delay at the beginning
        rng (int): Random seed. Default is None
    """

    def __init__(
        self,
        manifest_path=None,
        audio_tar_filepaths=None,
        shuffle_n=128,
        normalize_impulse=False,
        shift_impulse=False,
        rng=None,
    ):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)
        self._audiodataset = None
        self._tarred_audio = False
        self._normalize_impulse = normalize_impulse
        self._shift_impulse = shift_impulse
        self._data_iterator = None

        if audio_tar_filepaths:
            self._tarred_audio = True
            self._audiodataset = AugmentationDataset(manifest_path, audio_tar_filepaths, shuffle_n)
            self._data_iterator = iter(self._audiodataset)

        self._rng = rng
        random.seed(self._rng) if rng else None

    def perturb(self, data):
        impulse = read_one_audiosegment(
            self._manifest, data.sample_rate, tarred_audio=self._tarred_audio, audio_dataset=self._data_iterator,
        )

        # normalize if necessary
        if self._normalize_impulse:
            # normalize the impulse response to zero mean and amplitude 1
            impulse_norm = impulse.samples - np.mean(impulse.samples)
            impulse_norm /= max(abs(impulse_norm))
        else:
            impulse_norm = impulse.samples

        # len of input data samples
        len_data = len(data._samples)

        # convolve with the full impulse response
        data._samples = signal.fftconvolve(data._samples, impulse_norm, "full")

        # compensate the dominant path propagation delay
        if self._shift_impulse:
            # Find the peak of the IR and shift the output to the left
            max_ind = np.argmax(np.abs(impulse_norm))
            data._samples = data._samples[max_ind:]

        # trim to match the input data length
        data._samples = data._samples[:len_data]

        # normalize data samples to [-1,1] after rir convolution to avoid nans with fp16 training
        data._samples = data._samples / max(abs(data._samples))


class ShiftPerturbation(Perturbation):
    """
    Perturbs audio by shifting the audio in time by a random amount between min_shift_ms and max_shift_ms.
    The final length of the audio is kept unaltered by padding the audio with zeros.


    Args:
        min_shift_ms (float): Minimum time in milliseconds by which audio will be shifted
        max_shift_ms (float): Maximum time in milliseconds by which audio will be shifted
        rng (int): Random seed. Default is None
    """

    def __init__(self, min_shift_ms=-5.0, max_shift_ms=5.0, rng=None):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        random.seed(rng) if rng else None

    def perturb(self, data):
        shift_ms = random.uniform(self._min_shift_ms, self._max_shift_ms)
        if abs(shift_ms) / 1000 > data.duration:
            # TODO: do something smarter than just ignore this condition
            return
        shift_samples = int(shift_ms * data.sample_rate // 1000)
        # logging.debug("shift: %s", shift_samples)
        if shift_samples < 0:
            data._samples[-shift_samples:] = data._samples[:shift_samples]
            data._samples[:-shift_samples] = 0
        elif shift_samples > 0:
            data._samples[:-shift_samples] = data._samples[shift_samples:]
            data._samples[-shift_samples:] = 0


class NoisePerturbation(Perturbation):
    """
    Perturbation that adds noise to input audio.

    Args:
        manifest_path (str): Manifest file with paths to noise files
        min_snr_db (float): Minimum SNR of audio after noise is added
        max_snr_db (float): Maximum SNR of audio after noise is added
        max_gain_db (float): Maximum gain that can be applied on the noise sample
        audio_tar_filepaths (list) : Tar files, if noise audio files are tarred
        shuffle_n (int): Shuffle parameter for shuffling buffered files from the tar files
        orig_sr (int): Original sampling rate of the noise files
        rng (int): Random seed. Default is None
    """

    def __init__(
        self,
        manifest_path=None,
        min_snr_db=10,
        max_snr_db=50,
        max_gain_db=300.0,
        rng=None,
        audio_tar_filepaths=None,
        shuffle_n=100,
        orig_sr=16000,
    ):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)
        self._audiodataset = None
        self._tarred_audio = False
        self._orig_sr = orig_sr
        self._data_iterator = None

        if audio_tar_filepaths:
            self._tarred_audio = True
            self._audiodataset = AugmentationDataset(manifest_path, audio_tar_filepaths, shuffle_n)
            self._data_iterator = iter(self._audiodataset)

        random.seed(rng) if rng else None
        self._rng = rng

        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    @property
    def orig_sr(self):
        return self._orig_sr

    def get_one_noise_sample(self, target_sr):
        return read_one_audiosegment(
            self._manifest, target_sr, tarred_audio=self._tarred_audio, audio_dataset=self._data_iterator
        )

    def perturb(self, data, ref_mic=0):
        """
        Args:
            data (AudioSegment): audio data
            ref_mic (int): reference mic index for scaling multi-channel audios
        """
        noise = read_one_audiosegment(
            self._manifest, data.sample_rate, tarred_audio=self._tarred_audio, audio_dataset=self._data_iterator,
        )
        self.perturb_with_input_noise(data, noise, ref_mic=ref_mic)

    def perturb_with_input_noise(self, data, noise, data_rms=None, ref_mic=0):
        """
        Args:
            data (AudioSegment): audio data
            noise (AudioSegment): noise data
            data_rms (Union[float, List[float]): rms_db for data input
            ref_mic (int): reference mic index for scaling multi-channel audios
        """
        if data.num_channels != noise.num_channels:
            raise ValueError(
                f"Found mismatched channels for data ({data.num_channels}) and noise ({noise.num_channels})."
            )

        if not (0 <= ref_mic < data.num_channels):
            raise ValueError(
                f" reference mic ID must be an integer in [0, {data.num_channels}), got {ref_mic} instead."
            )

        snr_db = random.uniform(self._min_snr_db, self._max_snr_db)
        if data_rms is None:
            data_rms = data.rms_db

        if data.num_channels > 1:
            noise_gain_db = data_rms[ref_mic] - noise.rms_db[ref_mic] - snr_db
        else:
            noise_gain_db = data_rms - noise.rms_db - snr_db
        noise_gain_db = min(noise_gain_db, self._max_gain_db)

        # calculate noise segment to use
        start_time = random.uniform(0.0, noise.duration - data.duration)
        if noise.duration > (start_time + data.duration):
            noise.subsegment(start_time=start_time, end_time=start_time + data.duration)

        # adjust gain for snr purposes and superimpose
        noise.gain_db(noise_gain_db)

        if noise._samples.shape[0] < data._samples.shape[0]:
            noise_idx = random.randint(0, data._samples.shape[0] - noise._samples.shape[0])
            data._samples[noise_idx : noise_idx + noise._samples.shape[0]] += noise._samples

        else:
            data._samples += noise._samples

    def perturb_with_foreground_noise(self, data, noise, data_rms=None, max_noise_dur=2, max_additions=1, ref_mic=0):
        """
        Args:
            data (AudioSegment): audio data
            noise (AudioSegment): noise data
            data_rms (Union[float, List[float]): rms_db for data input
            max_noise_dur: (float): max noise duration
            max_additions (int): number of times for adding noise
            ref_mic (int): reference mic index for scaling multi-channel audios
        """
        if data.num_channels != noise.num_channels:
            raise ValueError(
                f"Found mismatched channels for data ({data.num_channels}) and noise ({noise.num_channels})."
            )

        if not (0 <= ref_mic < data.num_channels):
            raise ValueError(
                f" reference mic ID must be an integer in [0, {data.num_channels}), got {ref_mic} instead."
            )

        snr_db = random.uniform(self._min_snr_db, self._max_snr_db)
        if not data_rms:
            data_rms = data.rms_db

        if data.num_channels > 1:
            noise_gain_db = data_rms[ref_mic] - noise.rms_db[ref_mic] - snr_db
        else:
            noise_gain_db = data_rms - noise.rms_db - snr_db
        noise_gain_db = min(noise_gain_db, self._max_gain_db)

        n_additions = random.randint(1, max_additions)

        for i in range(n_additions):
            noise_dur = random.uniform(0.0, max_noise_dur)
            start_time = random.uniform(0.0, noise.duration)
            start_sample = int(round(start_time * noise.sample_rate))
            end_sample = int(round(min(noise.duration, (start_time + noise_dur)) * noise.sample_rate))
            noise_samples = np.copy(noise._samples[start_sample:end_sample])
            # adjust gain for snr purposes and superimpose
            noise_samples *= 10.0 ** (noise_gain_db / 20.0)

            if noise_samples.shape[0] > data._samples.shape[0]:
                noise_samples = noise_samples[0 : data._samples.shape[0]]

            noise_idx = random.randint(0, data._samples.shape[0] - noise_samples.shape[0])
            data._samples[noise_idx : noise_idx + noise_samples.shape[0]] += noise_samples


class NoisePerturbationWithNormalization(Perturbation):
    """
    Perturbation that adds noise to input audio, with normalisation to specific decibel level.
    Also tiles shorter noise samples up to their corresponding clean audio length.

    Args:
        manifest_path (str or list): Manifest file with paths to noise files, can be list if using multiple noise sources
        min_snr_db (float): Minimum SNR of audio after noise is added
        max_snr_db (float): Maximum SNR of audio after noise is added
        snr_samples (list): A discrete list of SNRs DBs to sample from when mixing, will be used instead of [min_snr_db,max_snr_db]
        norm_to_db (float): Will normalise clean, noise, and mixed samples to this DB
        audio_tar_filepaths (str or list) : Tar files, if noise audio files are tarred, can be list for multiple sources
        shuffle_n (int): Shuffle parameter for shuffling buffered files from the tar files
        orig_sr (int): Original sampling rate of the noise files
        rng (int): Random seed. Default is None
        shard_strategy (str): if you're using tarred audio and wish to scatter instead of replicate, set this to 'scatter'
        epsilon (float): minimum value for RMS DB normalisation to avoid divide by zero
    """

    def __init__(
        self,
        manifest_path=None,
        min_snr_db=10,
        max_snr_db=50,
        snr_samples=None,
        norm_to_db=None,
        rng=None,
        audio_tar_filepaths=None,
        shuffle_n=128,
        orig_sr=16000,
        global_rank=0,
        world_size=1,
        shard_strategy='replicate',
        epsilon=0.01,
    ):
        # import here to avoid circular import error
        from nemo.collections.asr.data.audio_to_text import RandomizedChainDataset

        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)
        self._audiodataset = None
        self._tarred_audio = False
        self._orig_sr = orig_sr
        self._data_iterator = None

        random.seed(rng) if rng else None
        self._rng = rng

        if audio_tar_filepaths:
            self._tarred_audio = True
            if isinstance(manifest_path, str):
                manifest_path = [manifest_path]
            if isinstance(audio_tar_filepaths, str):
                audio_tar_filepaths = [audio_tar_filepaths]
            datasets = []
            for tarred_audio_filepath, manifest_filepath in zip(audio_tar_filepaths, manifest_path):
                dataset = AugmentationDataset(
                    manifest_filepath,
                    tarred_audio_filepath,
                    shuffle_n,
                    rank=global_rank,
                    world_size=world_size,
                    shard_strategy=shard_strategy,
                )
                datasets.append(dataset)
            self._audiodataset = RandomizedChainDataset(
                datasets, rnd_seed=(rng if rng else random.randint(0, 30000)) + global_rank
            )
            if len(self._audiodataset) == 0:
                raise RuntimeError(
                    "NoisePerturbationWithNormalization detected a zero length RandomizedChainDataset, should never happen"
                )
            self._data_iterator = iter(self._audiodataset)

        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._norm_to_db = norm_to_db
        self._snr_samples = snr_samples if isinstance(snr_samples, list) and len(snr_samples) > 0 else None
        self._epsilon = epsilon

    @property
    def orig_sr(self):
        return self._orig_sr

    def read_one_audiosegment(self, target_sr):
        if self._tarred_audio:
            if self._data_iterator is None:
                raise TypeError("Expected valid iterator but got None")
            try:
                audio_file, file_id, manifest_entry = next(self._data_iterator)
            except StopIteration:
                self._data_iterator = iter(self._audiodataset)
                audio_file, file_id, manifest_entry = next(self._data_iterator)

            offset = 0 if manifest_entry.offset is None else manifest_entry.offset
            duration = 0 if manifest_entry.duration is None else manifest_entry.duration

        else:
            audio_record = random.sample(self._manifest.data, 1)[0]
            audio_file = audio_record.audio_file
            offset = 0 if audio_record.offset is None else audio_record.offset
            duration = 0 if audio_record.duration is None else audio_record.duration

        return AudioSegment.from_file(audio_file, target_sr=target_sr, offset=offset, duration=duration)

    def perturb(self, data, ref_mic=0):
        """
        Args:
            data (AudioSegment): audio data
            ref_mic (int): reference mic index for scaling multi-channel audios
        """

        noise = self.read_one_audiosegment(data.sample_rate)

        # noise samples need to be at least 1 second long to avoid strange oddities
        # in the RMS SNR mixing, so we have a fail-safe here to ensure at least 1 sec duration
        while noise.duration < 1:
            noise = self.read_one_audiosegment(data.sample_rate)

        self.perturb_with_input_noise(data, noise, ref_mic=ref_mic, norm_to_db=self._norm_to_db)

    def snr_mixer(self, clean, noise, snr, norm_to_db=-25.0):
        """
        Mixes the clean audio with the noise
        Args:
            clean (numpy array): the clean audio data
            noise (numpy array): the noise audio data
            snr (float): the SNR value for the mixing
            norm_to_db (float): the DB value to normalise to before mixing
        """
        clean = self.norm_audio_to_db(clean, norm_to_db)
        noise = self.norm_audio_to_db(noise, norm_to_db)

        # Set the noise level for a given SNR
        # note that if your noise doesn't overlap with your audio then your target SNR
        # may not be achievable. Consider using an rms-threshold in the future
        noisescalar = 10 ** (-snr / 20.0)
        noisenewlevel = noise * noisescalar
        noisyspeech = clean + noisenewlevel

        return clean, noisenewlevel, noisyspeech

    def norm_audio_to_db(self, x, norm_to_db):
        """
        Normalises audio signal to particular db, with some epsilon in-case of divide by zero
        Args:
            x (numpy array): input audio signal
            norm_to_db (float): the db to normalise to
        """
        rms = (x ** 2).mean(axis=0) ** 0.5
        rms = np.where(np.isclose(rms, 0), self._epsilon, rms)
        scalar = 10 ** (norm_to_db / 20.0) / rms
        return x * scalar

    def concatenate_noise_sample(self, clean, noise, fs, silence_length=0.25):
        """
        Tiles the noise array to match the clean audio array, with small silence between the joins
        Args:
            clean (numpy array): clean audio data
            noise (numpy array): noise audio data
            fs (int): sample rate used by both clean and noise audio data
            silence_length (float): the amount of silence (in secs) to insert before tiling
        """
        while len(noise) < len(clean):
            if noise.ndim > 1:
                zeros = np.zeros((int(fs * silence_length), noise.shape[-1]))
            else:
                zeros = np.zeros((int(fs * silence_length),))
            noiseconcat = np.append(noise, zeros, axis=0)
            noise = np.append(noiseconcat, noise, axis=0)

        return noise

    def perturb_with_input_noise(self, data, noise, data_rms=None, ref_mic=0, norm_to_db=-25.0):
        """
        Args:
            data (AudioSegment): audio data
            noise (AudioSegment): noise data
            data_rms (Union[float, List[float]): rms_db for data input
            ref_mic (int): reference mic index for scaling multi-channel audio, if set to None then
                           each channel will be scaled independently
            norm_to_db (float): will normalise all audio to this DB
        """
        if data.num_channels != noise.num_channels:
            raise ValueError(
                f"Found mismatched channels for data ({data.num_channels}) and noise ({noise.num_channels})."
            )

        if not (0 <= ref_mic < data.num_channels):
            raise ValueError(
                f" reference mic ID must be an integer in [0, {data.num_channels}), got {ref_mic} instead."
            )

        if self._snr_samples:
            snr_db = random.sample(self._snr_samples, 1)[0]
        else:
            snr_db = random.uniform(self._min_snr_db, self._max_snr_db)
        if data_rms is None:
            data_rms = data.rms_db[ref_mic] if isinstance(data.rms_db, (list, np.ndarray)) else data.rms_db

        if norm_to_db is None:
            norm_to_db = data_rms

        data_norm = data._samples
        noise_norm = noise._samples

        if len(data_norm) == 0:
            return

        if len(noise_norm) < len(data_norm):
            noise_norm = self.concatenate_noise_sample(data_norm, noise_norm, data.sample_rate)
        noise_norm = noise_norm[0 : len(data_norm)]

        _, _, noisy_snr = self.snr_mixer(clean=data_norm, noise=noise_norm, snr=snr_db, norm_to_db=norm_to_db)

        data._samples = noisy_snr


class WhiteNoisePerturbation(Perturbation):
    """
    Perturbation that adds white noise to an audio file in the training dataset.

    Args:
        min_level (int): Minimum level in dB at which white noise should be added
        max_level (int): Maximum level in dB at which white noise should be added
        rng (int): Random seed. Default is None
    """

    def __init__(self, min_level=-90, max_level=-46, rng=None):
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        np.random.seed(rng) if rng else None

    def perturb(self, data):
        noise_level_db = np.random.randint(self.min_level, self.max_level, dtype='int32')
        noise_signal = np.random.randn(data._samples.shape[0]) * (10.0 ** (noise_level_db / 20.0))
        data._samples += noise_signal


class RirAndNoisePerturbation(Perturbation):
    """
        RIR augmentation with additive foreground and background noise.
        In this implementation audio data is augmented by first convolving the audio with a Room Impulse Response
        and then adding foreground noise and background noise at various SNRs. RIR, foreground and background noises
        should either be supplied with a manifest file or as tarred audio files (faster).

        Different sets of noise audio files based on the original sampling rate of the noise. This is useful while
        training a mixed sample rate model. For example, when training a mixed model with 8 kHz and 16 kHz audio with a
        target sampling rate of 16 kHz, one would want to augment 8 kHz data with 8 kHz noise rather than 16 kHz noise.

        Args:
            rir_manifest_path: Manifest file for RIRs
            rir_tar_filepaths: Tar files, if RIR audio files are tarred
            rir_prob: Probability of applying a RIR
            noise_manifest_paths: Foreground noise manifest path
            min_snr_db: Min SNR for foreground noise
            max_snr_db: Max SNR for background noise,
            noise_tar_filepaths: Tar files, if noise files are tarred
            apply_noise_rir: Whether to convolve foreground noise with a a random RIR
            orig_sample_rate: Original sampling rate of foreground noise audio
            max_additions: Max number of times foreground noise is added to an utterance,
            max_duration: Max duration of foreground noise
            bg_noise_manifest_paths: Background noise manifest path
            bg_min_snr_db: Min SNR for background noise
            bg_max_snr_db: Max SNR for background noise
            bg_noise_tar_filepaths: Tar files, if noise files are tarred
            bg_orig_sample_rate: Original sampling rate of background noise audio
            rng: Random seed. Default is None

    """

    def __init__(
        self,
        rir_manifest_path=None,
        rir_prob=0.5,
        noise_manifest_paths=None,
        noise_prob=1.0,
        min_snr_db=0,
        max_snr_db=50,
        rir_tar_filepaths=None,
        rir_shuffle_n=100,
        noise_tar_filepaths=None,
        apply_noise_rir=False,
        orig_sample_rate=None,
        max_additions=5,
        max_duration=2.0,
        bg_noise_manifest_paths=None,
        bg_noise_prob=1.0,
        bg_min_snr_db=10,
        bg_max_snr_db=50,
        bg_noise_tar_filepaths=None,
        bg_orig_sample_rate=None,
        rng=None,
    ):

        self._rir_prob = rir_prob
        self._noise_prob = noise_prob
        self._bg_noise_prob = bg_noise_prob
        random.seed(rng) if rng else None
        self._rir_perturber = ImpulsePerturbation(
            manifest_path=rir_manifest_path,
            audio_tar_filepaths=rir_tar_filepaths,
            shuffle_n=rir_shuffle_n,
            shift_impulse=True,
        )
        self._fg_noise_perturbers = None
        self._bg_noise_perturbers = None
        if noise_manifest_paths:
            self._fg_noise_perturbers = {}
            for i in range(len(noise_manifest_paths)):
                if orig_sample_rate is None:
                    orig_sr = 16000
                else:
                    orig_sr = orig_sample_rate[i]
                self._fg_noise_perturbers[orig_sr] = NoisePerturbation(
                    manifest_path=noise_manifest_paths[i],
                    min_snr_db=min_snr_db[i],
                    max_snr_db=max_snr_db[i],
                    audio_tar_filepaths=noise_tar_filepaths[i],
                    orig_sr=orig_sr,
                )
        self._max_additions = max_additions
        self._max_duration = max_duration
        if bg_noise_manifest_paths:
            self._bg_noise_perturbers = {}
            for i in range(len(bg_noise_manifest_paths)):
                if bg_orig_sample_rate is None:
                    orig_sr = 16000
                else:
                    orig_sr = bg_orig_sample_rate[i]
                self._bg_noise_perturbers[orig_sr] = NoisePerturbation(
                    manifest_path=bg_noise_manifest_paths[i],
                    min_snr_db=bg_min_snr_db[i],
                    max_snr_db=bg_max_snr_db[i],
                    audio_tar_filepaths=bg_noise_tar_filepaths[i],
                    orig_sr=orig_sr,
                )

        self._apply_noise_rir = apply_noise_rir

    def perturb(self, data):
        prob = random.uniform(0.0, 1.0)

        if prob < self._rir_prob:
            self._rir_perturber.perturb(data)

        data_rms = data.rms_db

        if self._fg_noise_perturbers is not None and random.uniform(0.0, 1.0) < self._noise_prob:
            orig_sr = data.orig_sr
            if orig_sr not in self._fg_noise_perturbers:
                orig_sr = max(self._fg_noise_perturbers.keys())
            fg_perturber = self._fg_noise_perturbers[orig_sr]
            noise = fg_perturber.get_one_noise_sample(data.sample_rate)
            if self._apply_noise_rir:
                self._rir_perturber.perturb(noise)
            fg_perturber.perturb_with_foreground_noise(
                data, noise, data_rms=data_rms, max_noise_dur=self._max_duration, max_additions=self._max_additions
            )

        if self._bg_noise_perturbers is not None and random.uniform(0.0, 1.0) < self._bg_noise_prob:
            orig_sr = data.orig_sr
            if orig_sr not in self._bg_noise_perturbers:
                orig_sr = max(self._bg_noise_perturbers.keys())
            bg_perturber = self._bg_noise_perturbers[orig_sr]

            noise = bg_perturber.get_one_noise_sample(data.sample_rate)
            bg_perturber.perturb_with_input_noise(data, noise, data_rms=data_rms)


class TranscodePerturbation(Perturbation):
    """
        Audio codec augmentation. This implementation uses sox to transcode audio with low rate audio codecs,
        so users need to make sure that the installed sox version supports the codecs used here (G711 and amr-nb).

        Args:
            codecs (List[str]):A list of codecs to be trancoded to. Default is None.
            rng (int): Random seed. Default is None.
    """

    def __init__(self, codecs=None, rng=None):
        random.seed(rng) if rng else None
        self._codecs = codecs if codecs is not None else ["g711", "amr-nb", "ogg"]
        self.att_factor = 0.8  # to avoid saturation while writing to wav
        if codecs is not None:
            for codec in codecs:
                if codec not in ["g711", "amr-nb", "ogg"]:
                    raise ValueError(
                        f"TranscodePerturbation with {codec} isnot supported. Only {codecs} are supported"
                    )

    def perturb(self, data):
        max_level = np.max(np.abs(data._samples))
        if max_level > 0.8:
            norm_factor = self.att_factor / max_level
            norm_samples = norm_factor * data._samples
        else:
            norm_samples = data._samples
        orig_f = NamedTemporaryFile(suffix=".wav")
        sf.write(orig_f.name, norm_samples.transpose(), 16000)

        codec_ind = random.randint(0, len(self._codecs) - 1)
        if self._codecs[codec_ind] == "amr-nb":
            transcoded_f = NamedTemporaryFile(suffix="_amr.wav")
            rates = list(range(0, 4))
            rate = rates[random.randint(0, len(rates) - 1)]
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0 -C {rate} -t amr-nb - | sox -t amr-nb - -V0 -b 16 -r 16000 {transcoded_f.name}",
                shell=True,
            )
        elif self._codecs[codec_ind] == "ogg":
            transcoded_f = NamedTemporaryFile(suffix="_ogg.wav")
            rates = list(range(-1, 8))
            rate = rates[random.randint(0, len(rates) - 1)]
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0 -C {rate} -t ogg - | sox -t ogg - -V0 -b 16 -r 16000 {transcoded_f.name}",
                shell=True,
            )
        elif self._codecs[codec_ind] == "g711":
            transcoded_f = NamedTemporaryFile(suffix="_g711.wav")
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0  -r 8000 -c 1 -e a-law {transcoded_f.name} lowpass 3400 highpass 300",
                shell=True,
            )

        new_data = AudioSegment.from_file(transcoded_f.name, target_sr=16000)
        data._samples = new_data._samples[0 : data._samples.shape[0]]
        return


class RandomSegmentPerturbation(Perturbation):
    """
    Returns a random segment from input of duration "duration_sec". 
    If duration_sec > input audio length, pad_to_duration determines the outcome.
    
    RandomSegmentPerturbation is intended for self-supervised learning.
    Not for supervised, as extracting corresponding text is not facilitated.


    Args:
        duration_sec (float): duration of the segment to be extracted
        pad_to_duration (bool): zero pad if length of input audio < duration_sec
        rng: Random seed. Default is None
    """

    def __init__(self, duration_sec=32.0, pad_to_duration=False, rng=None):
        if duration_sec <= 0:
            raise ValueError("duration_sec should be > 0")

        self._duration_sec = duration_sec
        self._pad_to_duration = pad_to_duration
        random.seed(rng) if rng else None

    def perturb(self, data):
        if self._duration_sec > data.duration:
            if not self._pad_to_duration:
                raise ValueError(f"audio length < {self._duration_sec} sec and pad_to_duration is set to False")
            start_time = 0.0
            pad_size = self._duration_sec * data.sample_rate - data.num_samples
            data.pad(pad_size=pad_size)
        else:
            start_time = random.uniform(0.0, data.duration - self._duration_sec)

        end_time = start_time + self._duration_sec
        data.subsegment(start_time=start_time, end_time=end_time)


perturbation_types = {
    "speed": SpeedPerturbation,
    "time_stretch": TimeStretchPerturbation,
    "gain": GainPerturbation,
    "silence": SilencePerturbation,
    "impulse": ImpulsePerturbation,
    "shift": ShiftPerturbation,
    "noise": NoisePerturbation,
    "noise_norm": NoisePerturbationWithNormalization,
    "white_noise": WhiteNoisePerturbation,
    "rir_noise_aug": RirAndNoisePerturbation,
    "transcode_aug": TranscodePerturbation,
    "random_segment": RandomSegmentPerturbation,
}


def register_perturbation(name: str, perturbation: Perturbation):
    if name in perturbation_types.keys():
        raise KeyError(
            f"Perturbation with the name {name} exists. " f"Type of perturbation : {perturbation_types[name]}."
        )

    perturbation_types[name] = perturbation


class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        random.seed(rng) if rng else None
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, segment):
        for (prob, p) in self._pipeline:
            if random.random() < prob:
                p.perturb(segment)
        return

    def max_augmentation_length(self, length):
        newlen = length
        for (prob, p) in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen

    @classmethod
    def from_config(cls, config):
        ptbs = []
        for p in config:
            if p['aug_type'] not in perturbation_types:
                logging.warning("%s perturbation not known. Skipping.", p['aug_type'])
                continue
            perturbation = perturbation_types[p['aug_type']]
            ptbs.append((p['prob'], perturbation(**p['cfg'])))
        return cls(perturbations=ptbs)


def process_augmentations(augmenter, global_rank=0, world_size=1) -> Optional[AudioAugmentor]:
    """Process list of online data augmentations.
    Accepts either an AudioAugmentor object with pre-defined augmentations,
    or a dictionary that points to augmentations that have been defined.
    If a dictionary is passed, must follow the below structure:
    Dict[str, Dict[str, Any]]: Which refers to a dictionary of string
    names for augmentations, defined in `asr/parts/perturb.py`.
    The inner dictionary may contain key-value arguments of the specific
    augmentation, along with an essential key `prob`. `prob` declares the
    probability of the augmentation being applied, and must be a float
    value in the range [0, 1].
    # Example in YAML config file
    Augmentations are generally applied only during training, so we can add
    these augmentations to our yaml config file, and modify the behaviour
    for training and evaluation.
    ```yaml
    AudioToSpeechLabelDataLayer:
        ...  # Parameters shared between train and evaluation time
        train:
            augmentor:
                shift:
                    prob: 0.5
                    min_shift_ms: -5.0
                    max_shift_ms: 5.0
                white_noise:
                    prob: 1.0
                    min_level: -90
                    max_level: -46
                ...
        eval:
            ...
    ```
    Then in the training script,
    ```python
    import copy
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(model_config) as f:
        params = yaml.load(f)
    # Train Config for Data Loader
    train_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    train_dl_params.update(params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]
    data_layer_train = nemo_asr.AudioToTextDataLayer(
        ...,
        **train_dl_params,
    )
    # Evaluation Config for Data Loader
    eval_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    eval_dl_params.update(params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]
    data_layer_eval = nemo_asr.AudioToTextDataLayer(
        ...,
        **eval_dl_params,
    )
    ```
    # Registering your own Augmentations
    To register custom augmentations to obtain the above convenience of
    the declaring the augmentations in YAML, you can put additional keys in
    `perturbation_types` dictionary as follows.
    ```python
    from nemo.collections.asr.parts import perturb
    # Define your own perturbation here
    class CustomPerturbation(perturb.Perturbation):
        ...
    perturb.register_perturbation(name_of_perturbation, CustomPerturbation)
    ```
    Args:
        augmenter: AudioAugmentor object or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
    Returns: AudioAugmentor object
    """
    if augmenter is None:
        return None

    if isinstance(augmenter, AudioAugmentor):
        return augmenter

    augmenter_types = {dict}
    if HAVE_OMEGACONG_WEBDATASET:
        augmenter_types = {dict, DictConfig}
    if not type(augmenter) in augmenter_types:
        raise ValueError("Cannot parse augmenter. Must be a dict or an AudioAugmentor object ")

    if HAVE_OMEGACONG_WEBDATASET and isinstance(augmenter, DictConfig):
        augmenter = OmegaConf.to_container(augmenter, resolve=True)

    augmenter = copy.deepcopy(augmenter)

    augmentations = []
    for augment_name, augment_kwargs in augmenter.items():
        prob = augment_kwargs.get('prob', None)

        if prob is None:
            raise KeyError(
                f'Augmentation "{augment_name}" will not be applied as '
                f'keyword argument "prob" was not defined for this augmentation.'
            )

        else:
            _ = augment_kwargs.pop('prob')

            if prob < 0.0 or prob > 1.0:
                raise ValueError("`prob` must be a float value between 0 and 1.")

            try:
                augmentation_class = perturbation_types[augment_name]
                if 'global_rank' in inspect.signature(augmentation_class).parameters:
                    augment_kwargs['global_rank'] = global_rank
                if 'world_size' in inspect.signature(augmentation_class).parameters:
                    augment_kwargs['world_size'] = world_size
                augmentation = augmentation_class(**augment_kwargs)
                augmentations.append([prob, augmentation])
            except KeyError:
                raise KeyError(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")

    augmenter = AudioAugmentor(perturbations=augmentations)
    return augmenter


class AugmentationDataset(IterableDataset):
    """
        A class that loads tarred audio files and cycles over the files in the dataset.
        Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
        as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
        contain the information for one audio file, including at least the transcript and name of the audio
        file within the tarball.
        Valid formats for the audio_tar_filepaths argument include:
        (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
        (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].
        Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
        This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
        Supported opening braces - { <=> (, [, < and the special tag _OP_.
        Supported closing braces - } <=> ), ], > and the special tag _CL_.
        For SLURM based tasks, we suggest the use of the special tags for ease of use.
        See the WebDataset documentation for more information about accepted data and input formats.
    """

    def __init__(
        self,
        manifest_path: str,
        tar_filepaths: Union[str, List[str]],
        shuffle_n: int = 128,
        rank: int = 0,
        world_size: int = 1,
        shard_strategy: str = "replicate",
    ):
        # import here to avoid circular import error
        from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths

        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)

        tar_filepaths = expand_sharded_filepaths(
            tar_filepaths, shard_strategy=shard_strategy, world_size=world_size, global_rank=rank
        )

        if not HAVE_OMEGACONG_WEBDATASET:
            raise LightningNotInstalledException(self)
        self.audio_dataset = wd.WebDataset(urls=tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self.audio_dataset = self.audio_dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self.audio_dataset = (
            self.audio_dataset.rename(audio='wav;ogg;flac', key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._loop_offsets)
        )

    def __len__(self):
        return len(self._manifest)

    def _loop_offsets(self, iterator):
        """This function is used to iterate through utterances with different offsets for each file.
        """

        class TarredAudioLoopOffsets:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection
                self.current_fn = None
                self.current_bytes = None
                self.offset_id = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.current_fn is None:
                    self.current_bytes, self.current_fn = next(self.iterator)
                    self.offset_id = 0
                else:
                    offset_list = self.collection.mapping[self.current_fn]
                    if len(offset_list) == self.offset_id + 1:
                        self.current_bytes, self.current_fn = next(self.iterator)
                        self.offset_id = 0
                    else:
                        self.offset_id += 1

                return self.current_bytes, self.current_fn, self.offset_id

        return TarredAudioLoopOffsets(self._manifest)

    def __iter__(self):
        audio_iter = iter(self.audio_dataset)

        while True:
            try:
                audio_bytes, audio_filename, offset_id = next(audio_iter)
                file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                manifest_idx = self._manifest.mapping[file_id][offset_id]
                manifest_entry = self._manifest[manifest_idx]

                # Convert audio bytes to IO stream for processing (for SoundFile to read)
                audio_file = io.BytesIO(audio_bytes)
                yield audio_file, file_id, manifest_entry
            except StopIteration:
                audio_iter = iter(self.audio_dataset)
