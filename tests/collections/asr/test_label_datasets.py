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
import json
import os
import tempfile

import pytest
import torch

from nemo.collections.asr.data.audio_to_label import TarredAudioToClassificationLabelDataset
from nemo.collections.asr.data.feature_to_label import FeatureToLabelDataset, FeatureToSeqSpeakerLabelDataset
from nemo.collections.asr.parts.preprocessing.feature_loader import ExternalFeatureLoader
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer


class TestASRDatasets:
    labels = ["fash", "fbbh", "fclc"]
    unique_labels_in_seq = ['0', '1', '2', '3', "zero", "one", "two", "three"]

    @pytest.mark.unit
    def test_tarred_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/tarred_audio_manifest.json'))

        # Test braceexpand loading
        tarpath = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/audio_{0..1}.tar'))
        featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)
        ds_braceexpand = TarredAudioToClassificationLabelDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, featurizer=featurizer
        )

        assert len(ds_braceexpand) == 32
        count = 0
        for _ in ds_braceexpand:
            count += 1
        assert count == 32

        # Test loading via list
        tarpath = [os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/audio_{i}.tar')) for i in range(2)]
        ds_list_load = TarredAudioToClassificationLabelDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, featurizer=featurizer
        )
        count = 0
        for _ in ds_list_load:
            count += 1
        assert count == 32

    @pytest.mark.unit
    def test_tarred_dataset_duplicate_name(self, test_data_dir):
        manifest_path = os.path.abspath(
            os.path.join(test_data_dir, 'asr/tarred_an4/tarred_duplicate_audio_manifest.json')
        )

        # Test braceexpand loading
        tarpath = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/audio_{0..1}.tar'))
        featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)
        ds_braceexpand = TarredAudioToClassificationLabelDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, featurizer=featurizer
        )

        assert len(ds_braceexpand) == 6
        count = 0
        for _ in ds_braceexpand:
            count += 1
        assert count == 6

        # Test loading via list
        tarpath = [os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/audio_{i}.tar')) for i in range(2)]
        ds_list_load = TarredAudioToClassificationLabelDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, featurizer=featurizer
        )
        count = 0
        for _ in ds_list_load:
            count += 1
        assert count == 6

    @pytest.mark.unit
    def test_feat_seqlabel_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/feat/emb.json'))
        feature_loader = ExternalFeatureLoader(augmentor=None)
        ds_braceexpand = FeatureToSeqSpeakerLabelDataset(
            manifest_filepath=manifest_path, labels=self.unique_labels_in_seq, feature_loader=feature_loader
        )
        # fmt: off
        correct_label = torch.tensor(
            [0.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 2.0, 0.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0,]
        )
        # fmt: on
        correct_label_length = torch.tensor(50)

        assert ds_braceexpand[0][0].shape == (50, 32)
        assert torch.equal(ds_braceexpand[0][2], correct_label)
        assert torch.equal(ds_braceexpand[0][3], correct_label_length)

        count = 0
        for _ in ds_braceexpand:
            count += 1
        assert count == 2

    @pytest.mark.unit
    def test_feat_label_dataset(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest_input.json')
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                for i in range(2):
                    feat_file = os.path.join(tmpdir, f"feat_{i}.pt")
                    torch.save(torch.randn(80, 5), feat_file)
                    entry = {'feature_file': feat_file, 'duration': 100000, 'label': '0'}
                    fp.write(json.dumps(entry) + '\n')

            dataset = FeatureToLabelDataset(manifest_filepath=manifest_path, labels=self.unique_labels_in_seq)

            correct_label = torch.tensor(self.unique_labels_in_seq.index('0'))
            correct_label_length = torch.tensor(1)

            assert dataset[0][0].shape == (80, 5)
            assert torch.equal(dataset[0][2], correct_label)
            assert torch.equal(dataset[0][3], correct_label_length)

            count = 0
            for _ in dataset:
                count += 1
            assert count == 2
