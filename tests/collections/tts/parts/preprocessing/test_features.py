# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import contextlib
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo.collections.tts.parts.preprocessing.features import (
    EnergyFeaturizer,
    MelSpectrogramFeaturizer,
    PitchFeaturizer,
)


class TestTTSFeatures:
    def setup_class(self):
        self.audio_filename = "test.wav"
        self.spec_dim = 80
        self.hop_len = 100
        self.audio_len = 50000
        self.sample_rate = 10000
        self.spec_len = 1 + (self.audio_len // self.hop_len)
        self.manifest_entry = {"audio_filepath": self.audio_filename}

    def _compute_start_end_frames(self, offset, duration):
        start_frame = int(self.sample_rate * offset // self.hop_len)
        end_frame = 1 + int(self.sample_rate * (offset + duration) // self.hop_len)
        return start_frame, end_frame

    @contextlib.contextmanager
    def _create_test_dir(self):
        test_audio = np.random.uniform(size=[self.audio_len])
        temp_dir = tempfile.TemporaryDirectory()
        try:
            test_dir = Path(temp_dir.name)
            audio_path = test_dir / self.audio_filename
            sf.write(audio_path, test_audio, self.sample_rate)
            yield test_dir
        finally:
            temp_dir.cleanup()

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_compute_mel_spectrogram(self):
        mel_featurizer = MelSpectrogramFeaturizer(
            mel_dim=self.spec_dim, hop_length=self.hop_len, sample_rate=self.sample_rate
        )

        with self._create_test_dir() as test_dir:
            spec = mel_featurizer.compute_mel_spec(manifest_entry=self.manifest_entry, audio_dir=test_dir)

        assert len(spec.shape) == 2
        assert spec.dtype == np.float32
        assert spec.shape[0] == self.spec_dim
        assert spec.shape[1] == self.spec_len

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_save_and_load_mel_spectrogram(self):
        mel_name = "mel_test"
        mel_featurizer = MelSpectrogramFeaturizer(
            feature_name=mel_name, mel_dim=self.spec_dim, hop_length=self.hop_len, sample_rate=self.sample_rate
        )

        with self._create_test_dir() as test_dir:
            feature_dir = test_dir / "feature"
            mel_featurizer.save(manifest_entry=self.manifest_entry, audio_dir=test_dir, feature_dir=feature_dir)
            mel_dict = mel_featurizer.load(
                manifest_entry=self.manifest_entry, audio_dir=test_dir, feature_dir=feature_dir
            )

        mel_spec = mel_dict[mel_name]
        assert len(mel_spec.shape) == 2
        assert mel_spec.dtype == torch.float32
        assert mel_spec.shape[0] == self.spec_dim
        assert mel_spec.shape[1] == self.spec_len

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_compute_pitch(self):
        pitch_featurizer = PitchFeaturizer(hop_length=self.hop_len, sample_rate=self.sample_rate)

        with self._create_test_dir() as test_dir:
            pitch, voiced, voiced_prob = pitch_featurizer.compute_pitch(
                manifest_entry=self.manifest_entry, audio_dir=test_dir
            )

        assert len(pitch.shape) == 1
        assert pitch.shape[0] == self.spec_len
        assert pitch.dtype == np.float32

        assert len(voiced.shape) == 1
        assert voiced.shape[0] == self.spec_len
        assert voiced.dtype == bool

        assert len(voiced_prob.shape) == 1
        assert voiced_prob.shape[0] == self.spec_len
        assert voiced_prob.dtype == np.float32

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_compute_pitch_batched(self, test_data_dir):
        audio_filepath = Path(test_data_dir) / "tts" / "mini_ljspeech" / "wavs" / "LJ003-0182.wav"
        manifest_entry = {"audio_filepath": audio_filepath}

        # Compute pitch with batching disabled
        pitch_featurizer = PitchFeaturizer(hop_length=self.hop_len, sample_rate=self.sample_rate, batch_seconds=None)
        pitch, voiced, voiced_prob = pitch_featurizer.compute_pitch(
            manifest_entry=manifest_entry, audio_dir=test_data_dir
        )

        # Compute pitch in 1 second chunks
        pitch_featurizer_batch = PitchFeaturizer(
            hop_length=self.hop_len, sample_rate=self.sample_rate, batch_seconds=1.0
        )
        pitch_batch, voiced_batch, voiced_prob_batch = pitch_featurizer_batch.compute_pitch(
            manifest_entry=manifest_entry, audio_dir=test_data_dir
        )

        torch.testing.assert_close(pitch_batch, pitch)
        torch.testing.assert_close(voiced_batch, voiced)
        torch.testing.assert_close(voiced_prob_batch, voiced_prob)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_save_and_load_pitch(self):
        pitch_name = "pitch_test"
        voiced_mask_name = "voiced_mask_test"
        voiced_prob_name = "voiced_prob_test"
        pitch_featurizer = PitchFeaturizer(
            pitch_name=pitch_name,
            voiced_mask_name=voiced_mask_name,
            voiced_prob_name=voiced_prob_name,
            hop_length=self.hop_len,
            sample_rate=self.sample_rate,
        )

        with self._create_test_dir() as test_dir:
            feature_dir = test_dir / "feature"
            pitch_featurizer.save(manifest_entry=self.manifest_entry, audio_dir=test_dir, feature_dir=feature_dir)
            pitch_dict = pitch_featurizer.load(
                manifest_entry=self.manifest_entry, audio_dir=test_dir, feature_dir=feature_dir
            )

        pitch = pitch_dict[pitch_name]
        voiced_mask = pitch_dict[voiced_mask_name]
        voiced_prob = pitch_dict[voiced_prob_name]

        assert len(pitch.shape) == 1
        assert pitch.shape[0] == self.spec_len
        assert pitch.dtype == torch.float32

        assert len(voiced_mask.shape) == 1
        assert voiced_mask.shape[0] == self.spec_len
        assert voiced_mask.dtype == torch.bool

        assert len(voiced_prob.shape) == 1
        assert voiced_prob.shape[0] == self.spec_len
        assert voiced_prob.dtype == torch.float32

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_save_and_load_pitch_segments(self, test_data_dir):
        pitch_name = "pitch_test"
        voiced_mask_name = "voiced_mask_test"
        voiced_prob_name = "voiced_prob_test"
        audio_filepath = Path(test_data_dir) / "tts" / "mini_ljspeech" / "wavs" / "LJ003-0182.wav"
        manifest_entry = {"audio_filepath": audio_filepath}

        offset1 = 1.0
        duration1 = 1.0
        offset2 = 2.5
        duration2 = 1.2
        start1, end1 = self._compute_start_end_frames(offset=offset1, duration=duration1)
        start2, end2 = self._compute_start_end_frames(offset=offset2, duration=duration2)
        manifest_entry_segment1 = {"audio_filepath": audio_filepath, "offset": offset1, "duration": duration1}
        manifest_entry_segment2 = {"audio_filepath": audio_filepath, "offset": offset2, "duration": duration2}

        pitch_featurizer = PitchFeaturizer(
            pitch_name=pitch_name,
            voiced_mask_name=voiced_mask_name,
            voiced_prob_name=voiced_prob_name,
            hop_length=self.hop_len,
            sample_rate=self.sample_rate,
        )

        with self._create_test_dir() as test_dir:
            feature_dir = test_dir / "feature"
            pitch_featurizer.save(manifest_entry=manifest_entry, audio_dir=test_data_dir, feature_dir=feature_dir)
            pitch_dict = pitch_featurizer.load(
                manifest_entry=manifest_entry, audio_dir=test_data_dir, feature_dir=feature_dir
            )
            pitch_dict_segment1 = pitch_featurizer.load(
                manifest_entry=manifest_entry_segment1, audio_dir=test_data_dir, feature_dir=feature_dir
            )
            pitch_dict_segment2 = pitch_featurizer.load(
                manifest_entry=manifest_entry_segment2, audio_dir=test_data_dir, feature_dir=feature_dir
            )

        pitch = pitch_dict[pitch_name]
        voiced_mask = pitch_dict[voiced_mask_name]
        voiced_prob = pitch_dict[voiced_prob_name]

        pitch_segment1 = pitch_dict_segment1[pitch_name]
        voiced_mask_segment1 = pitch_dict_segment1[voiced_mask_name]
        voiced_prob_segment1 = pitch_dict_segment1[voiced_prob_name]

        pitch_segment2 = pitch_dict_segment2[pitch_name]
        voiced_mask_segment2 = pitch_dict_segment2[voiced_mask_name]
        voiced_prob_segment2 = pitch_dict_segment2[voiced_prob_name]

        torch.testing.assert_close(pitch_segment1, pitch[start1:end1])
        torch.testing.assert_close(voiced_mask_segment1, voiced_mask[start1:end1])
        torch.testing.assert_close(voiced_prob_segment1, voiced_prob[start1:end1])

        torch.testing.assert_close(pitch_segment2, pitch[start2:end2])
        torch.testing.assert_close(voiced_mask_segment2, voiced_mask[start2:end2])
        torch.testing.assert_close(voiced_prob_segment2, voiced_prob[start2:end2])

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_compute_energy(self):
        mel_featurizer = MelSpectrogramFeaturizer(
            mel_dim=self.spec_dim, hop_length=self.hop_len, sample_rate=self.sample_rate
        )
        energy_featurizer = EnergyFeaturizer(spec_featurizer=mel_featurizer)

        with self._create_test_dir() as test_dir:
            energy = energy_featurizer.compute_energy(manifest_entry=self.manifest_entry, audio_dir=test_dir)

        assert len(energy.shape) == 1
        assert energy.shape[0] == self.spec_len
        assert energy.dtype == np.float32

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_save_and_load_energy(self):
        energy_name = "energy_test"
        mel_featurizer = MelSpectrogramFeaturizer(
            mel_dim=self.spec_dim, hop_length=self.hop_len, sample_rate=self.sample_rate
        )
        energy_featurizer = EnergyFeaturizer(feature_name=energy_name, spec_featurizer=mel_featurizer)

        with self._create_test_dir() as test_dir:
            feature_dir = test_dir / "feature"
            energy_featurizer.save(manifest_entry=self.manifest_entry, audio_dir=test_dir, feature_dir=feature_dir)
            energy_dict = energy_featurizer.load(
                manifest_entry=self.manifest_entry, audio_dir=test_dir, feature_dir=feature_dir
            )

        energy = energy_dict[energy_name]
        assert len(energy.shape) == 1
        assert energy.shape[0] == self.spec_len
        assert energy.dtype == torch.float32

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_save_and_load_energy_segments(self, test_data_dir):
        energy_name = "energy_test"
        offset1 = 0.0
        duration1 = 1.0
        offset2 = 1.0
        duration2 = 1.5
        start1, end1 = self._compute_start_end_frames(offset=offset1, duration=duration1)
        start2, end2 = self._compute_start_end_frames(offset=offset2, duration=duration2)
        manifest_entry_segment1 = {"audio_filepath": self.audio_filename, "offset": offset1, "duration": duration1}
        manifest_entry_segment2 = {"audio_filepath": self.audio_filename, "offset": offset2, "duration": duration2}

        mel_featurizer = MelSpectrogramFeaturizer(
            mel_dim=self.spec_dim, hop_length=self.hop_len, sample_rate=self.sample_rate
        )
        energy_featurizer = EnergyFeaturizer(feature_name=energy_name, spec_featurizer=mel_featurizer)

        with self._create_test_dir() as test_dir:
            feature_dir = test_dir / "feature"
            energy_featurizer.save(manifest_entry=self.manifest_entry, audio_dir=test_dir, feature_dir=feature_dir)
            energy_dict = energy_featurizer.load(
                manifest_entry=self.manifest_entry, audio_dir=test_dir, feature_dir=feature_dir
            )
            energy_dict_segment1 = energy_featurizer.load(
                manifest_entry=manifest_entry_segment1, audio_dir=test_dir, feature_dir=feature_dir
            )
            energy_dict_segment2 = energy_featurizer.load(
                manifest_entry=manifest_entry_segment2, audio_dir=test_dir, feature_dir=feature_dir
            )

        energy = energy_dict[energy_name]
        energy_segment1 = energy_dict_segment1[energy_name]
        energy_segment2 = energy_dict_segment2[energy_name]

        torch.testing.assert_close(energy_segment1, energy[start1:end1])
        torch.testing.assert_close(energy_segment2, energy[start2:end2])
