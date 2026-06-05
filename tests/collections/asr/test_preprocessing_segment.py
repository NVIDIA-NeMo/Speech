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

import json
import os
import tempfile
from collections import namedtuple
from typing import List, Type, Union

import numpy as np
import pytest
import soundfile as sf

from nemo.collections.asr.parts.preprocessing.perturb import NoisePerturbation, ShiftPerturbation, SilencePerturbation
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment, select_channels


class TestSelectChannels:
    num_samples = 1000
    max_diff_tol = 1e-9

    @pytest.mark.unit
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0, 1, [0, 1]])
    def test_single_channel_input(self, channel_selector: Type[Union[str, int, List[int]]]):
        """Cover the case with single-channel input signal.
        Channel selector should not do anything in this case.
        """
        golden_out = signal_in = np.random.rand(self.num_samples)

        if channel_selector not in [None, 0, 'average']:
            # Expect a failure if looking for a different channel when input is 1D
            with pytest.raises(ValueError):
                # UUT
                select_channels(signal_in, channel_selector)
        else:
            # UUT
            signal_out = select_channels(signal_in, channel_selector)

            # Check difference
            max_diff = np.max(np.abs(signal_out - golden_out))
            assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [2, 4])
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0, [1], [0, 1]])
    def test_multi_channel_input(self, num_channels: int, channel_selector: Type[Union[str, int, List[int]]]):
        """Cover the case with multi-channel input signal and single-
        or multi-channel output.
        """
        signal_in = np.random.rand(self.num_samples, num_channels)

        # calculate golden output
        if channel_selector is None:
            golden_out = signal_in
        elif channel_selector == 'average':
            golden_out = np.mean(signal_in, axis=1)
        else:
            golden_out = signal_in[:, channel_selector].squeeze()

        # UUT
        signal_out = select_channels(signal_in, channel_selector)

        # Check difference
        max_diff = np.max(np.abs(signal_out - golden_out))
        assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [1, 2])
    @pytest.mark.parametrize("channel_selector", [2, [1, 2]])
    def test_select_more_channels_than_available(
        self, num_channels: int, channel_selector: Type[Union[str, int, List[int]]]
    ):
        """This test is expecting the UUT to fail because we ask for more channels
        than available in the input signal.
        """
        signal_in = np.random.rand(self.num_samples, num_channels)

        # expect failure since we ask for more channels than available
        with pytest.raises(ValueError):
            # UUT
            select_channels(signal_in, channel_selector)


class TestAudioSegment:

    sample_rate = 16000
    signal_duration_sec = 2
    max_diff_tol = 1e-9

    @property
    def num_samples(self):
        return self.sample_rate * self.signal_duration_sec

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [1, 4])
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0, 1, [0, 1]])
    def test_init_single_channel(self, num_channels: int, channel_selector: Type[Union[str, int, List[int]]]):
        """Test the constructor directly."""
        if num_channels == 1:
            # samples is a one-dimensional vector for single-channel signal
            samples = np.random.rand(self.num_samples)
        else:
            samples = np.random.rand(self.num_samples, num_channels)

        if (isinstance(channel_selector, int) and channel_selector >= num_channels) or (
            isinstance(channel_selector, list) and max(channel_selector) >= num_channels
        ):
            # Expect a failure if looking for a different channel when input is 1D
            with pytest.raises(ValueError):
                # Construct UUT
                uut = AudioSegment(samples=samples, sample_rate=self.sample_rate, channel_selector=channel_selector)
        else:
            # Construct UUT
            uut = AudioSegment(samples=samples, sample_rate=self.sample_rate, channel_selector=channel_selector)

            # Create golden reference
            # Note: AudioSegment converts input samples to float32
            golden_samples = select_channels(samples.astype('float32'), channel_selector)
            expected_num_channels = 1 if golden_samples.ndim == 1 else golden_samples.shape[1]

            # Test UUT
            assert uut.num_channels == expected_num_channels
            assert uut.num_samples == self.num_samples
            assert uut.sample_rate == self.sample_rate
            assert uut.duration == self.signal_duration_sec
            max_diff = np.max(np.abs(uut.samples - golden_samples))
            assert max_diff < self.max_diff_tol

            # Test zero padding
            pad_length = 42
            uut.pad(pad_length, symmetric=False)
            # compare to golden references
            assert uut.num_samples == self.num_samples + pad_length
            assert np.all(uut.samples[-pad_length:] == 0.0)
            max_diff = np.max(np.abs(uut.samples[:-pad_length] - golden_samples))
            assert max_diff < self.max_diff_tol

            # Test subsegment
            start_time = 0.2 * self.signal_duration_sec
            end_time = 0.5 * self.signal_duration_sec
            uut.subsegment(start_time=start_time, end_time=end_time)
            # compare to golden references
            start_sample = int(round(start_time * self.sample_rate))
            end_sample = int(round(end_time * self.sample_rate))
            max_diff = np.max(np.abs(uut.samples - golden_samples[start_sample:end_sample]))
            assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [1, 4])
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0])
    def test_from_file(self, num_channels, channel_selector):
        """Test loading a signal from a file."""
        with tempfile.TemporaryDirectory() as test_dir:
            # Prepare a wav file
            audio_file = os.path.join(test_dir, 'audio.wav')
            if num_channels == 1:
                # samples is a one-dimensional vector for single-channel signal
                samples = np.random.rand(self.num_samples)
            else:
                samples = np.random.rand(self.num_samples, num_channels)
            sf.write(audio_file, samples, self.sample_rate, 'float')

            # Create UUT
            uut = AudioSegment.from_file(audio_file, channel_selector=channel_selector)

            # Create golden reference
            # Note: AudioSegment converts input samples to float32
            golden_samples = select_channels(samples.astype('float32'), channel_selector)
            expected_num_channels = 1 if golden_samples.ndim == 1 else golden_samples.shape[1]

            # Test UUT
            assert uut.num_channels == expected_num_channels
            assert uut.num_samples == self.num_samples
            assert uut.sample_rate == self.sample_rate
            assert uut.duration == self.signal_duration_sec
            max_diff = np.max(np.abs(uut.samples - golden_samples))
            assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("data_channels", [1, 4])
    @pytest.mark.parametrize("noise_channels", [1, 4])
    def test_noise_perturb_channels(self, data_channels, noise_channels):
        """Test loading a signal from a file."""
        with tempfile.TemporaryDirectory() as test_dir:
            # Prepare a wav file
            audio_file = os.path.join(test_dir, 'audio.wav')
            if data_channels == 1:
                # samples is a one-dimensional vector for single-channel signal
                samples = np.random.rand(self.num_samples)
            else:
                samples = np.random.rand(self.num_samples, data_channels)
            sf.write(audio_file, samples, self.sample_rate, 'float')

            noise_file = os.path.join(test_dir, 'noise.wav')
            if noise_channels == 1:
                # samples is a one-dimensional vector for single-channel signal
                samples = np.random.rand(self.num_samples)
            else:
                samples = np.random.rand(self.num_samples, noise_channels)
            sf.write(noise_file, samples, self.sample_rate, 'float')

            manifest_file = os.path.join(test_dir, 'noise_manifest.json')
            with open(manifest_file, 'w') as fout:
                item = {'audio_filepath': os.path.abspath(noise_file), 'label': '-', 'duration': 0.1, 'offset': 0.0}
                fout.write(f'{json.dumps(item)}\n')

            perturber = NoisePerturbation(manifest_file)
            audio = AudioSegment.from_file(audio_file)
            noise = AudioSegment.from_file(noise_file)

            if data_channels == noise_channels:
                try:
                    _ = perturber.perturb_with_input_noise(audio, noise, ref_mic=0)
                except ValueError as e:
                    assert False, "perturb_with_input_noise failed with ref_mic=0"

                with pytest.raises(ValueError):
                    _ = perturber.perturb_with_input_noise(audio, noise, ref_mic=data_channels)

                try:
                    _ = perturber.perturb_with_foreground_noise(audio, noise, ref_mic=0)
                except ValueError as e:
                    assert False, "perturb_with_foreground_noise failed with ref_mic=0"

                with pytest.raises(ValueError):
                    _ = perturber.perturb_with_foreground_noise(audio, noise, ref_mic=data_channels)
            else:
                with pytest.raises(ValueError):
                    _ = perturber.perturb_with_input_noise(audio, noise)
                with pytest.raises(ValueError):
                    _ = perturber.perturb_with_foreground_noise(audio, noise)

    def test_silence_perturb(self):
        """Test loading a signal from a file and apply silence perturbation"""
        with tempfile.TemporaryDirectory() as test_dir:
            # Prepare a wav file
            audio_file = os.path.join(test_dir, 'audio.wav')
            # samples is a one-dimensional vector for single-channel signal
            samples = np.random.rand(self.num_samples)
            sf.write(audio_file, samples, self.sample_rate, 'float')

            dur = 2
            perturber = SilencePerturbation(
                min_start_silence_secs=dur,
                max_start_silence_secs=dur,
                min_end_silence_secs=dur,
                max_end_silence_secs=dur,
            )

            audio = AudioSegment.from_file(audio_file)
            ori_audio_len = len(audio._samples)
            _ = perturber.perturb(audio)

            assert len(audio._samples) == ori_audio_len + 2 * dur * self.sample_rate

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_channels, channel_selectors",
        [
            (1, [None, 'average', 0]),
            (3, [None, 'average', 0, 1, [0, 1]]),
        ],
    )
    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22500])
    def test_audio_segment_from_file(self, tmpdir, num_channels, channel_selectors, sample_rate):
        """Test loading and audio signal from a file."""
        signal_len_sec = 4
        num_samples = signal_len_sec * sample_rate
        num_examples = 10
        rtol, atol = 1e-5, 1e-6

        for n in range(num_examples):
            # Create a test vector
            audio_file = os.path.join(tmpdir, f'test_audio_{n:02}.wav')
            samples = np.random.randn(num_samples, num_channels)
            sf.write(audio_file, samples, sample_rate, 'float')

            for channel_selector in channel_selectors:
                if channel_selector is None:
                    ref_samples = samples
                elif isinstance(channel_selector, int) or isinstance(channel_selector, list):
                    ref_samples = samples[:, channel_selector]
                elif channel_selector == 'average':
                    ref_samples = np.mean(samples, axis=1)
                else:
                    raise ValueError(f'Unexpected value of channel_selector {channel_selector}')

                # 1) Load complete audio
                # Reference
                ref_samples = ref_samples.squeeze()
                ref_channels = 1 if ref_samples.ndim == 1 else ref_samples.shape[1]

                # UUT
                audio_segment = AudioSegment.from_file(audio_file, channel_selector=channel_selector)

                # Test
                assert (
                    audio_segment.sample_rate == sample_rate
                ), f'channel_selector {channel_selector}, sample rate not matching: {audio_segment.sample_rate} != {sample_rate}'
                assert (
                    audio_segment.num_channels == ref_channels
                ), f'channel_selector {channel_selector}, num channels not matching: {audio_segment.num_channels} != {ref_channels}'
                assert audio_segment.num_samples == len(
                    ref_samples
                ), f'channel_selector {channel_selector}, num samples not matching: {audio_segment.num_samples} != {len(ref_samples)}'
                assert np.allclose(
                    audio_segment.samples, ref_samples, rtol=rtol, atol=atol
                ), f'channel_selector {channel_selector}, samples not matching'

                # 2) Load a with duration=None and offset=None, should load the whole audio

                # UUT
                audio_segment = AudioSegment.from_file(
                    audio_file, offset=None, duration=None, channel_selector=channel_selector
                )

                # Test
                assert (
                    audio_segment.sample_rate == sample_rate
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, sample rate not matching: {audio_segment.sample_rate} != {sample_rate}'
                assert (
                    audio_segment.num_channels == ref_channels
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, num channels not matching: {audio_segment.num_channels} != {ref_channels}'
                assert audio_segment.num_samples == len(
                    ref_samples
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, num samples not matching: {audio_segment.num_samples} != {len(ref_samples)}'
                assert np.allclose(
                    audio_segment.samples, ref_samples, rtol=rtol, atol=atol
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, samples not matching'

                # 3) Load a random segment
                offset = 0.45 * np.random.rand() * signal_len_sec
                duration = 0.45 * np.random.rand() * signal_len_sec

                # Reference
                start = int(offset * sample_rate)
                end = start + int(duration * sample_rate)
                ref_samples = ref_samples[start:end, ...]

                # UUT
                audio_segment = AudioSegment.from_file(
                    audio_file, offset=offset, duration=duration, channel_selector=channel_selector
                )

                # Test
                assert (
                    audio_segment.sample_rate == sample_rate
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, sample rate not matching: {audio_segment.sample_rate} != {sample_rate}'
                assert (
                    audio_segment.num_channels == ref_channels
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, num channels not matching: {audio_segment.num_channels} != {ref_channels}'
                assert audio_segment.num_samples == len(
                    ref_samples
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, num samples not matching: {audio_segment.num_samples} != {len(ref_samples)}'
                assert np.allclose(
                    audio_segment.samples, ref_samples, rtol=rtol, atol=atol
                ), f'channel_selector {channel_selector}, offset {offset}, duration {duration}, samples not matching'

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_channels, channel_selectors",
        [
            (1, [None, 'average', 0]),
            (3, [None, 'average', 0, 1, [0, 1]]),
        ],
    )
    @pytest.mark.parametrize("offset", [0, 1.5])
    @pytest.mark.parametrize("duration", [1, 2])
    def test_audio_segment_multichannel_with_list(self, tmpdir, num_channels, channel_selectors, offset, duration):
        """Test loading an audio signal from a list of single-channel files."""
        sample_rate = 16000
        signal_len_sec = 5
        num_samples = signal_len_sec * sample_rate
        rtol, atol = 1e-5, 1e-6

        # Random samples
        samples = np.random.rand(num_samples, num_channels)

        # Save audio
        audio_files = []
        for m in range(num_channels):
            a_file = os.path.join(tmpdir, f'ch_{m}.wav')
            sf.write(a_file, samples[:, m], sample_rate)
            audio_files.append(a_file)
        mc_file = os.path.join(tmpdir, f'mc.wav')
        sf.write(mc_file, samples, sample_rate)

        for channel_selector in channel_selectors:

            # UUT: loading audio from a list of files
            uut_segment = AudioSegment.from_file(
                audio_file=audio_files, offset=offset, duration=duration, channel_selector=channel_selector
            )

            # Reference: load from the original file
            ref_segment = AudioSegment.from_file(
                audio_file=mc_file, offset=offset, duration=duration, channel_selector=channel_selector
            )

            # Check
            assert (
                uut_segment.sample_rate == ref_segment.sample_rate
            ), f'channel_selector {channel_selector}: expecting {ref_segment.sample_rate}, but UUT segment has {uut_segment.sample_rate}'
            assert (
                uut_segment.num_samples == ref_segment.num_samples
            ), f'channel_selector {channel_selector}: expecting {ref_segment.num_samples}, but UUT segment has {uut_segment.num_samples}'
            assert np.allclose(
                uut_segment.samples, ref_segment.samples, rtol=rtol, atol=atol
            ), f'channel_selector {channel_selector}: samples not matching'

        # Try to get a channel that is out of range.
        with pytest.raises(RuntimeError, match="Channel cannot be selected"):
            AudioSegment.from_file(audio_file=audio_files, channel_selector=num_channels)

        if num_channels > 1:
            # Try to load a list of multichannel files
            # This is expected to fail since we only support loading a single-channel signal
            # from each file when audio_file is a list
            with pytest.raises(RuntimeError, match="Expecting a single-channel audio signal"):
                AudioSegment.from_file(audio_file=[mc_file, mc_file])

            with pytest.raises(RuntimeError, match="Expecting a single-channel audio signal"):
                AudioSegment.from_file(audio_file=[mc_file, mc_file], channel_selector=0)

    @pytest.mark.unit
    @pytest.mark.parametrize("target_sr", [8000, 16000])
    def test_audio_segment_trim_match(self, tmpdir, target_sr):
        """Test loading and audio signal from a file matches when using a path and a list
        for different target_sr, int_values and trim setups.
        """
        sample_rate = 24000
        signal_len_sec = 2
        num_samples = signal_len_sec * sample_rate
        num_examples = 10

        TrimSetup = namedtuple("TrimSetup", "ref top_db frame_length hop_length")
        trim_setups = []
        trim_setups.append(TrimSetup(np.max, 10, 2048, 1024))
        trim_setups.append(TrimSetup(1.0, 35, 2048, 1024))
        trim_setups.append(TrimSetup(0.8, 45, 2048, 1024))

        for n in range(num_examples):
            # Create a test vector
            audio_file = os.path.join(tmpdir, f'test_audio_{n:02}.wav')
            samples = np.random.randn(num_samples)
            # normalize
            samples = samples / np.max(samples)
            # apply random scaling and window to have some samples cut by trim
            samples = np.random.rand() * np.hanning(num_samples) * samples
            sf.write(audio_file, samples, sample_rate, 'float')

            for trim_setup in trim_setups:
                # UUT 1: load from a path
                audio_segment_1 = AudioSegment.from_file(
                    audio_file,
                    target_sr=target_sr,
                    trim=True,
                    trim_ref=trim_setup.ref,
                    trim_top_db=trim_setup.top_db,
                    trim_frame_length=trim_setup.frame_length,
                    trim_hop_length=trim_setup.hop_length,
                )

                # UUT 2: load from a list
                audio_segment_2 = AudioSegment.from_file(
                    [audio_file],
                    target_sr=target_sr,
                    trim=True,
                    trim_ref=trim_setup.ref,
                    trim_top_db=trim_setup.top_db,
                    trim_frame_length=trim_setup.frame_length,
                    trim_hop_length=trim_setup.hop_length,
                )

                # Test
                assert audio_segment_1 == audio_segment_2, f'trim setup {trim_setup}, loaded segments not matching'


class TestShiftPerturbation:
    sample_rate = 16000

    def _make_audio_segment(self, duration_sec=1.0):
        """Create a simple AudioSegment with a sine wave for testing."""
        num_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
        samples = np.sin(2 * np.pi * 440 * t)
        return AudioSegment(samples=samples, sample_rate=self.sample_rate)

    def test_shift_perturbation_normal(self):
        """Shift perturbation modifies audio when shift is within duration."""
        perturb = ShiftPerturbation(min_shift_ms=-5.0, max_shift_ms=5.0)
        segment = self._make_audio_segment(duration_sec=1.0)
        original = segment.samples.copy()
        perturb.perturb(segment)
        assert segment.samples.shape == original.shape, "Audio length should not change"

    def test_shift_perturbation_short_audio_not_skipped(self):
        """Shift perturbation should clamp and apply shift for short audio, not silently skip."""
        perturb = ShiftPerturbation(min_shift_ms=10.0, max_shift_ms=20.0)
        duration_sec = 0.005  # 5ms — shorter than min_shift_ms
        segment = self._make_audio_segment(duration_sec=duration_sec)
        original = segment.samples.copy()
        perturb.perturb(segment)
        assert segment.samples.shape == original.shape, "Audio length should not change"

    @pytest.mark.parametrize("duration_sec", [0.001, 0.01, 0.1, 1.0])
    def test_shift_perturbation_preserves_length(self, duration_sec):
        """Audio length must be preserved regardless of duration."""
        perturb = ShiftPerturbation(min_shift_ms=-50.0, max_shift_ms=50.0)
        segment = self._make_audio_segment(duration_sec=duration_sec)
        original_len = len(segment.samples)
        perturb.perturb(segment)
        assert len(segment.samples) == original_len, "Shift perturbation must preserve audio length"

    def test_shift_perturbation_zero_shift(self):
        """When min and max shift are both 0, audio should be unchanged."""
        perturb = ShiftPerturbation(min_shift_ms=0.0, max_shift_ms=0.0)
        segment = self._make_audio_segment(duration_sec=0.5)
        original = segment.samples.copy()
        perturb.perturb(segment)
        np.testing.assert_array_equal(segment.samples, original, "Zero shift should not modify audio")
