# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo.collections.asr.data.audio_to_label import AudioToMultiLabelDataset, TarredAudioToClassificationLabelDataset
from nemo.collections.asr.data.feature_to_label import (
    FeatureToLabelDataset,
    FeatureToMultiLabelDataset,
    FeatureToSeqSpeakerLabelDataset,
)
from nemo.collections.asr.parts.preprocessing.feature_loader import ExternalFeatureLoader
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common.parts.preprocessing.collections import ASRFeatureLabel, ASRSpeechLabel


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

    @staticmethod
    def _write_feature_manifest(tmpdir, labels, key='feature_file', name='manifest_input.json'):
        manifest_path = os.path.join(tmpdir, name)
        with open(manifest_path, 'w', encoding='utf-8') as fp:
            for i, label in enumerate(labels):
                feat_file = os.path.join(tmpdir, f"feat_{i}.pt")
                torch.save(torch.randn(80, 5), feat_file)
                fp.write(json.dumps({key: feat_file, 'duration': 1.0, 'label': label}) + '\n')
        return manifest_path

    @pytest.mark.unit
    def test_feature_label_collection_collects_whole_labels(self):
        """`uniq_labels` must hold the labels themselves, not the characters they are spelled with."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_feature_manifest(tmpdir, ['speech', 'background', 'speech'])

            collection = ASRFeatureLabel(manifests_files=[manifest_path])

            assert collection.uniq_labels == ['background', 'speech']

    @pytest.mark.unit
    def test_feature_label_collection_matches_audio_sibling(self):
        """Control: `ASRSpeechLabel` is the audio-side equivalent and already gets this right."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_manifest = self._write_feature_manifest(tmpdir, ['speech', 'background'])
            audio_manifest = self._write_feature_manifest(
                tmpdir, ['speech', 'background'], key='audio_filepath', name='manifest_audio.json'
            )

            feature_collection = ASRFeatureLabel(manifests_files=[feat_manifest])
            audio_collection = ASRSpeechLabel(manifests_files=[audio_manifest])

            assert feature_collection.uniq_labels == audio_collection.uniq_labels

    @pytest.mark.unit
    def test_feat_label_dataset_infers_labels_when_none(self):
        """`labels=None` is the documented fallback; it must build a usable label vocabulary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_feature_manifest(tmpdir, ['speech', 'background'])

            dataset = FeatureToLabelDataset(manifest_filepath=manifest_path, labels=None)

            assert dataset.labels == ['background', 'speech']
            assert dataset.num_classes == 2
            assert dataset.label2id == {'background': 0, 'speech': 1}
            assert torch.equal(dataset[0][2], torch.tensor(1))

    @pytest.mark.unit
    def test_feat_label_dataset_matches_multilabel_sibling_when_labels_none(self):
        """Control: `FeatureToMultiLabelDataset` derives the same vocabulary from the same manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_feature_manifest(tmpdir, ['speech', 'background'])

            dataset = FeatureToLabelDataset(manifest_filepath=manifest_path, labels=None)
            multi_label_dataset = FeatureToMultiLabelDataset(manifest_filepath=manifest_path, labels=None)

            assert dataset.labels == multi_label_dataset.labels

    @pytest.mark.unit
    def test_feature_label_collection_supports_regression_labels(self):
        """Regression labels are floats; iterating over them is not possible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_feature_manifest(tmpdir, ['0.5', '1.5'])

            collection = ASRFeatureLabel(manifests_files=[manifest_path], is_regression_task=True)

            assert collection.uniq_labels == [0.5, 1.5]

    @pytest.mark.unit
    def test_audio_multilabel_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest_input.json')
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                for i in range(2):
                    audio_file = os.path.join(tmpdir, f"audio_{i}.wav")
                    data = np.random.normal(0, 1, 16000 * 10)
                    sf.write(audio_file, data, 16000)
                    entry = {'audio_filepath': audio_file, 'duration': 10, 'label': '0 1 0 1'}
                    fp.write(json.dumps(entry) + '\n')

            dataset = AudioToMultiLabelDataset(manifest_filepath=manifest_path, sample_rate=16000, labels=['0', '1'])

            correct_label = torch.tensor([0, 1, 0, 1])
            correct_label_length = torch.tensor(4)

            assert dataset[0][0].shape == torch.tensor([0.1] * 160000).shape
            assert torch.equal(dataset[0][2], correct_label)
            assert torch.equal(dataset[0][3], correct_label_length)

            count = 0
            for _ in dataset:
                count += 1
            assert count == 2
