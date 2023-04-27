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
        self.audio_len = 10000
        self.sample_rate = 20000
        self.spec_len = 1 + (self.audio_len // self.hop_len)
        self.manifest_entry = {"audio_filepath": self.audio_filename}

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
        assert spec.dtype == torch.float32
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
        assert pitch.dtype == torch.float32

        assert len(voiced.shape) == 1
        assert voiced.shape[0] == self.spec_len
        assert voiced.dtype == torch.bool

        assert len(voiced_prob.shape) == 1
        assert voiced_prob.shape[0] == self.spec_len
        assert voiced_prob.dtype == torch.float32

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
    def test_compute_energy(self):
        mel_featurizer = MelSpectrogramFeaturizer(
            mel_dim=self.spec_dim, hop_length=self.hop_len, sample_rate=self.sample_rate
        )
        energy_featurizer = EnergyFeaturizer(spec_featurizer=mel_featurizer)

        with self._create_test_dir() as test_dir:
            energy = energy_featurizer.compute_energy(manifest_entry=self.manifest_entry, audio_dir=test_dir)

        assert len(energy.shape) == 1
        assert energy.shape[0] == self.spec_len
        assert energy.dtype == torch.float32

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
