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
import copy
import filecmp
import json
import os
import shutil
import tempfile
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
import torch.cuda
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nemo.collections.asr.data import audio_to_audio_dataset, audio_to_text_dataset
from nemo.collections.asr.data.audio_to_audio import (
    ASRAudioProcessor,
    AudioToTargetDataset,
    AudioToTargetWithEmbeddingDataset,
    AudioToTargetWithReferenceDataset,
    _audio_collate_fn,
)
from nemo.collections.asr.data.audio_to_text import (
    DataStoreObject,
    TarredAudioToBPEDataset,
    TarredAudioToCharDataset,
    cache_datastore_manifests,
)
from nemo.collections.asr.data.audio_to_text_dali import (
    __DALI_MINIMUM_VERSION__,
    AudioToBPEDALIDataset,
    AudioToCharDALIDataset,
    is_dali_supported,
)
from nemo.collections.asr.data.audio_to_text_dataset import inject_dataloader_value_from_model_config
from nemo.collections.asr.data.feature_to_text import FeatureToBPEDataset, FeatureToCharDataset
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.utils.audio_utils import get_segment_start
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.common import tokenizers
from nemo.utils import logging

try:
    HAVE_DALI = is_dali_supported(__DALI_MINIMUM_VERSION__)
except (ImportError, ModuleNotFoundError):
    HAVE_DALI = False


def decode_chars(tokens, token_length, mapping):
    text = []
    tokens = tokens.cpu().numpy()
    for idx in tokens:
        text_token = mapping[idx]
        text.append(text_token)

    text = text[:token_length]
    text = ''.join(text)
    return text


def decode_subwords(tokens, token_length, tokenizer: tokenizers.TokenizerSpec):
    tokens = tokens.cpu().numpy()
    tokens = tokens[:token_length]
    text = tokenizer.ids_to_text(tokens)
    return text


class TestASRDatasets:
    labels = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    @pytest.mark.unit
    def test_tarred_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/tarred_audio_manifest.json'))

        # Test braceexpand loading
        tarpath = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/audio_{0..1}.tar'))
        ds_braceexpand = TarredAudioToCharDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, sample_rate=16000
        )
        assert len(ds_braceexpand) == 32
        count = 0
        for _ in ds_braceexpand:
            count += 1
        assert count == 32

        # Test loading via list
        tarpath = [os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/audio_{i}.tar')) for i in range(2)]
        ds_list_load = TarredAudioToCharDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, sample_rate=16000
        )
        count = 0
        for _ in ds_list_load:
            count += 1
        assert count == 32

    @pytest.mark.unit
    def test_mismatch_in_model_dataloader_config(self, caplog):
        logging._logger.propagate = True
        caplog.set_level(logging.WARNING)

        model_cfg = OmegaConf.create(dict(labels=OmegaConf.create(["a", "b", "c"])))
        dataloader_cfg = OmegaConf.create(dict(labels=copy.deepcopy(self.labels)))

        inject_dataloader_value_from_model_config(model_cfg, dataloader_cfg, key='labels')

        assert (
            """`labels` is explicitly provided to the data loader, and is different from the `labels` provided at the model level config."""
            in caplog.text
        )

        logging._logger.propagate = False

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_tarred_bpe_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/tarred_audio_manifest.json'))

        tokenizer_path = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
        tokenizer = tokenizers.AutoTokenizer(pretrained_model_name='bert-base-cased', vocab_file=tokenizer_path)

        # Test braceexpand loading
        tarpath = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/audio_{0..1}.tar'))
        ds_braceexpand = TarredAudioToBPEDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, tokenizer=tokenizer, sample_rate=16000
        )
        assert len(ds_braceexpand) == 32
        count = 0
        for _ in ds_braceexpand:
            count += 1
        assert count == 32

        # Test loading via list
        tarpath = [os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/audio_{i}.tar')) for i in range(2)]
        ds_list_load = TarredAudioToBPEDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, tokenizer=tokenizer, sample_rate=16000
        )
        count = 0
        for _ in ds_list_load:
            count += 1
        assert count == 32

    @pytest.mark.skipif(not HAVE_DALI, reason="NVIDIA DALI is not installed or incompatible version")
    @pytest.mark.unit
    def test_dali_char_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/an4_val.json'))

        num_samples = 10
        batch_size = 2
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        texts = []

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r', encoding='utf-8') as m:
                for ix, line in enumerate(m):
                    if ix >= num_samples:
                        break

                    line = line.replace("tests/data/", "tests/.data/").replace("\n", "")
                    f.write(f"{line}\n")

                    data = json.loads(line)
                    texts.append(data['text'])

            f.seek(0)

            dataset = AudioToCharDALIDataset(
                manifest_filepath=f.name,
                device=device,
                batch_size=batch_size,
                labels=self.labels,
                max_duration=16.0,
                parser='en',
                shuffle=False,
            )

            assert len(dataset) == (num_samples // batch_size)  # num batches
            count = 0
            original_transcripts = []
            for batch in dataset:
                transcripts = batch[2]  # transcript index in DALIOutputs
                transcripts_lengths = batch[3]  # transcript length index in DALIOutputs
                transcripts = [
                    decode_chars(transcript, transcripts_length, mapping=self.labels)
                    for transcript, transcripts_length in zip(transcripts, transcripts_lengths)
                ]
                original_transcripts.extend(transcripts)
                count += len(transcripts)
            assert count == num_samples

            # Assert transcripts are correct
            for text, og_transcript in zip(texts, original_transcripts):
                assert text == og_transcript

            # Repeat, now with shuffle enabled
            f.seek(0)

            dataset = AudioToCharDALIDataset(
                manifest_filepath=f.name,
                device=device,
                batch_size=batch_size,
                labels=self.labels,
                max_duration=16.0,
                parser='en',
                shuffle=True,
            )

            assert len(dataset) == (num_samples // batch_size)  # num batches
            count = 0
            shuffled_transcripts = []
            for batch in dataset:
                transcripts = batch[2]  # transcript index in DALIOutputs
                transcripts_lengths = batch[3]  # transcript length index in DALIOutputs
                transcripts = [
                    decode_chars(transcript, transcripts_length, mapping=self.labels)
                    for transcript, transcripts_length in zip(transcripts, transcripts_lengths)
                ]
                shuffled_transcripts.extend(transcripts)
                count += len(transcripts)
            assert count == num_samples

            samples_changed = 0
            for orig, shuffled in zip(original_transcripts, shuffled_transcripts):
                if orig != shuffled:
                    samples_changed += 1
            assert samples_changed > 1  # assume after shuffling at least 1 sample was displaced

            for og_transcript, shuffled_transcript in zip(sorted(original_transcripts), sorted(shuffled_transcripts)):
                assert og_transcript == shuffled_transcript

    @pytest.mark.skipif(not HAVE_DALI, reason="NVIDIA DALI is not installed or incompatible version")
    @pytest.mark.unit
    def test_dali_bpe_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/an4_val.json'))

        num_samples = 10
        batch_size = 2
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        texts = []

        tokenizer_path = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
        tokenizer = tokenizers.AutoTokenizer(pretrained_model_name='bert-base-cased', vocab_file=tokenizer_path)

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r', encoding='utf-8') as m:
                for ix, line in enumerate(m):
                    if ix >= num_samples:
                        break

                    line = line.replace("tests/data/", "tests/.data/").replace("\n", "")
                    f.write(f"{line}\n")

                    data = json.loads(line)
                    texts.append(data['text'])

            f.seek(0)

            dataset = AudioToBPEDALIDataset(
                manifest_filepath=f.name,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
                max_duration=16.0,
                shuffle=False,
            )

            assert len(dataset) == (num_samples // batch_size)  # num batches
            count = 0
            original_transcripts = []
            for batch in dataset:
                transcripts = batch[2]  # transcript index in DALIOutputs
                transcripts_lengths = batch[3]  # transcript length index in DALIOutputs
                transcripts = [
                    decode_subwords(transcript, transcripts_length, tokenizer=tokenizer)
                    for transcript, transcripts_length in zip(transcripts, transcripts_lengths)
                ]
                original_transcripts.extend(transcripts)
                count += len(transcripts)
            assert count == num_samples

            # Assert transcripts are correct
            for text, og_transcript in zip(texts, original_transcripts):
                assert text == og_transcript

            # Repeat, now with shuffle enabled
            f.seek(0)

            dataset = AudioToBPEDALIDataset(
                manifest_filepath=f.name,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
                max_duration=16.0,
                shuffle=True,
            )

            assert len(dataset) == (num_samples // batch_size)  # num batches
            count = 0
            shuffled_transcripts = []
            for batch in dataset:
                transcripts = batch[2]  # transcript index in DALIOutputs
                transcripts_lengths = batch[3]  # transcript length index in DALIOutputs
                transcripts = [
                    decode_subwords(transcript, transcripts_length, tokenizer=tokenizer)
                    for transcript, transcripts_length in zip(transcripts, transcripts_lengths)
                ]
                shuffled_transcripts.extend(transcripts)
                count += len(transcripts)
            assert count == num_samples

            samples_changed = 0
            for orig, shuffled in zip(original_transcripts, shuffled_transcripts):
                if orig != shuffled:
                    samples_changed += 1
            assert samples_changed > 1  # assume after shuffling at least 1 sample was displaced

            for og_transcript, shuffled_transcript in zip(sorted(original_transcripts), sorted(shuffled_transcripts)):
                assert og_transcript == shuffled_transcript

    @pytest.mark.skipif(not HAVE_DALI, reason="NVIDIA DALI is not installed or incompatible version")
    @pytest.mark.unit
    def test_dali_char_vs_ref_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/an4_val.json'))

        num_samples = 10
        batch_size = 1
        texts = []

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r') as m:
                for ix, line in enumerate(m):
                    if ix >= num_samples:
                        break

                    line = line.replace("tests/data/", "tests/.data/").replace("\n", "")
                    f.write(f"{line}\n")

                    data = json.loads(line)
                    texts.append(data['text'])

            f.seek(0)

            preprocessor = {
                '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                'dither': 0.0,
            }
            preprocessor_cfg = DictConfig(preprocessor)

            dataset_cfg = {
                'manifest_filepath': f.name,
                'sample_rate': 16000,
                'labels': self.labels,
                'batch_size': batch_size,
                'trim_silence': False,
                'max_duration': 16.7,
                'shuffle': False,
                'is_tarred': False,
            }
            dali_dataset = audio_to_text_dataset.get_dali_char_dataset(
                config=dataset_cfg,
                shuffle=False,
                device_id=0,
                global_rank=0,
                world_size=1,
                preprocessor_cfg=preprocessor_cfg,
            )
            ref_dataset = audio_to_text_dataset.get_char_dataset(config=dataset_cfg,)
            ref_dataloader = DataLoader(
                dataset=ref_dataset,
                batch_size=batch_size,
                collate_fn=ref_dataset.collate_fn,
                drop_last=False,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            ref_preprocessor = EncDecCTCModel.from_config_dict(preprocessor_cfg)

            for ref_data, dali_data in zip(ref_dataloader, dali_dataset):
                ref_audio, ref_audio_len, _, _ = ref_data
                ref_features, ref_features_len = ref_preprocessor(input_signal=ref_audio, length=ref_audio_len)

                dali_features, dali_features_len, _, _ = dali_data

                a = ref_features.cpu().numpy()[:, :, :ref_features_len]
                b = dali_features.cpu().numpy()[:, :, :dali_features_len]

                err = np.abs(a - b)
                assert np.mean(err) < 0.0001
                assert np.max(err) < 0.01

    @pytest.mark.skipif(not HAVE_DALI, reason="NVIDIA DALI is not installed or incompatible version")
    @pytest.mark.unit
    def test_tarred_dali_char_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/tarred_audio_manifest.json'))
        audio_tar_filepaths = [
            os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/audio_{idx}.tar')) for idx in range(2)
        ]
        audio_tar_index_filepaths = [
            os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/dali_index/audio_{idx}.index'))
            for idx in range(2)
        ]

        batch_size = 8
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        texts = []

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            num_samples = 0
            with open(manifest_path, 'r') as m:
                num_samples = len(m.readlines())

            dataset = AudioToCharDALIDataset(
                manifest_filepath=manifest_path,
                audio_tar_filepaths=audio_tar_filepaths,
                audio_tar_index_filepaths=audio_tar_index_filepaths,
                device=device,
                batch_size=batch_size,
                labels=self.labels,
                max_duration=16.0,
                parser='en',
                shuffle=False,
            )

            assert len(dataset) == (num_samples // batch_size)  # num batches
            count = 0
            original_transcripts = []
            for batch in dataset:
                transcripts = batch[2]  # transcript index in DALIOutputs
                transcripts_lengths = batch[3]  # transcript length index in DALIOutputs
                transcripts = [
                    decode_chars(transcript, transcripts_length, mapping=self.labels)
                    for transcript, transcripts_length in zip(transcripts, transcripts_lengths)
                ]
                original_transcripts.extend(transcripts)
                count += len(transcripts)
            assert count == num_samples

            # Assert transcripts are correct
            for text, og_transcript in zip(texts, original_transcripts):
                assert text == og_transcript

            dataset = AudioToCharDALIDataset(
                manifest_filepath=manifest_path,  # f.name,
                audio_tar_filepaths=audio_tar_filepaths,
                audio_tar_index_filepaths=audio_tar_index_filepaths,
                device=device,
                batch_size=batch_size,
                labels=self.labels,
                max_duration=16.0,
                parser='en',
                shuffle=True,
            )

            assert len(dataset) == (num_samples // batch_size)  # num batches
            count = 0
            shuffled_transcripts = []
            for batch in dataset:
                transcripts = batch[2]  # transcript index in DALIOutputs
                transcripts_lengths = batch[3]  # transcript length index in DALIOutputs
                transcripts = [
                    decode_chars(transcript, transcripts_length, mapping=self.labels)
                    for transcript, transcripts_length in zip(transcripts, transcripts_lengths)
                ]
                shuffled_transcripts.extend(transcripts)
                count += len(transcripts)
            assert count == num_samples

            samples_changed = 0
            for orig, shuffled in zip(original_transcripts, shuffled_transcripts):
                if orig != shuffled:
                    samples_changed += 1
            assert samples_changed > 1  # assume after shuffling at least 1 sample was displaced

            for og_transcript, shuffled_transcript in zip(sorted(original_transcripts), sorted(shuffled_transcripts)):
                assert og_transcript == shuffled_transcript

    @pytest.mark.skipif(not HAVE_DALI, reason="NVIDIA DALI is not installed or incompatible version")
    @pytest.mark.unit
    def test_dali_tarred_char_vs_ref_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/tarred_audio_manifest.json'))
        audio_tar_filepaths = [
            os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/audio_{idx}.tar')) for idx in range(2)
        ]
        audio_tar_index_filepaths = [
            os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/dali_index/audio_{idx}.index'))
            for idx in range(2)
        ]

        batch_size = 8
        texts = []

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            num_samples = 0
            with open(manifest_path, 'r') as m:
                for ix, line in enumerate(m):
                    data = json.loads(line)
                    texts.append(data['text'])
                    num_samples = ix

            preprocessor = {
                '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                'dither': 0.0,
            }
            preprocessor_cfg = DictConfig(preprocessor)

            dataset_cfg = {
                'manifest_filepath': f.name,
                'tarred_audio_filepaths': audio_tar_filepaths,
                'tarred_audio_index_filepaths': audio_tar_index_filepaths,
                'sample_rate': 16000,
                'labels': self.labels,
                'batch_size': batch_size,
                'trim_silence': False,
                'max_duration': 16.7,
                'shuffle': False,
                'is_tarred': False,
            }
            dali_dataset = audio_to_text_dataset.get_dali_char_dataset(
                config=dataset_cfg,
                shuffle=False,
                device_id=0,
                global_rank=0,
                world_size=1,
                preprocessor_cfg=preprocessor_cfg,
            )
            ref_dataset = audio_to_text_dataset.get_tarred_dataset(
                config=dataset_cfg, shuffle_n=0, global_rank=0, world_size=1
            )
            ref_dataloader = DataLoader(
                dataset=ref_dataset,
                batch_size=batch_size,
                collate_fn=ref_dataset.collate_fn,
                drop_last=False,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            ref_preprocessor = EncDecCTCModel.from_config_dict(preprocessor_cfg)

            for ref_data, dali_data in zip(ref_dataloader, dali_dataset):
                ref_audio, ref_audio_len, _, _ = ref_data
                ref_features, ref_features_len = ref_preprocessor(input_signal=ref_audio, length=ref_audio_len)

                dali_features, dali_features_len, _, _ = dali_data

                a = ref_features.cpu().numpy()[:, :, :ref_features_len]
                b = dali_features.cpu().numpy()[:, :, :dali_features_len]

                err = np.abs(a - b)
                assert np.mean(err) < 0.0001
                assert np.max(err) < 0.01

    @pytest.mark.unit
    def test_feature_to_text_char_dataset(self):
        num_samples = 5
        golden_feat_shape = (80, 5)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest_input.json')
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                for i in range(num_samples):
                    feat_file = os.path.join(tmpdir, f"feat_{i}.pt")
                    torch.save(torch.randn(80, 5), feat_file)
                    entry = {'audio_filepath': "", 'feature_file': feat_file, 'duration': 100000, "text": "a b c"}
                    fp.write(json.dumps(entry) + '\n')

            dataset = FeatureToCharDataset(manifest_path, labels=self.labels)
            cnt = 0
            for item in dataset:
                cnt += 1
                feat = item[0]
                token_len = item[3]
                assert feat.shape == golden_feat_shape
                assert torch.equal(token_len, torch.tensor(5))
            assert cnt == num_samples

    @pytest.mark.unit
    def test_feature_to_text_bpe_dataset(self, test_data_dir):
        num_samples = 5
        golden_feat_shape = (80, 5)
        tokenizer_path = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
        tokenizer = tokenizers.AutoTokenizer(pretrained_model_name='bert-base-cased', vocab_file=tokenizer_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest_input.json')
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                for i in range(num_samples):
                    feat_file = os.path.join(tmpdir, f"feat_{i}.pt")
                    torch.save(torch.randn(80, 5), feat_file)
                    entry = {'audio_filepath': "", 'feature_file': feat_file, 'duration': 100000, "text": "a b c"}
                    fp.write(json.dumps(entry) + '\n')

            dataset = FeatureToBPEDataset(manifest_path, tokenizer=tokenizer)
            cnt = 0
            for item in dataset:
                cnt += 1
                feat = item[0]
                token_len = item[3]
                assert feat.shape == golden_feat_shape
                assert torch.equal(token_len, torch.tensor(5))
            assert cnt == num_samples

    @pytest.mark.unit
    def test_feature_with_rttm_to_text_char_dataset(self):
        num_samples = 2
        golden_feat_shape = (80, 10)
        sample = torch.ones(80, 10)
        masked_sample = sample * FeatureToCharDataset.ZERO_LEVEL_SPEC_DB_VAL
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest_input.json')
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                feat_file = os.path.join(tmpdir, f"feat_0.pt")
                torch.save(sample, feat_file)

                rttm_file = os.path.join(tmpdir, f"rttm_0.rttm")
                with open(rttm_file, "w") as fout:
                    fout.write(f"SPEAKER <NA> 1 0 1 <NA> <NA> speech <NA> <NA>\n")

                entry = {
                    'audio_filepath': "",
                    'feature_file': feat_file,
                    'rttm_file': rttm_file,
                    'duration': 100000,
                    "text": "a b c",
                }
                fp.write(json.dumps(entry) + '\n')

                # second sample where all frames are not masked
                feat_file = os.path.join(tmpdir, f"feat_1.pt")
                torch.save(sample, feat_file)

                rttm_file = os.path.join(tmpdir, f"rttm_1.rttm")
                with open(rttm_file, "w") as fout:
                    fout.write(f"SPEAKER <NA> 1 0 0 <NA> <NA> speech <NA> <NA>\n")

                entry = {
                    'audio_filepath': "",
                    'feature_file': feat_file,
                    'rttm_file': rttm_file,
                    'duration': 100000,
                    "text": "a b c",
                }
                fp.write(json.dumps(entry) + '\n')

            dataset = FeatureToCharDataset(manifest_path, labels=self.labels, normalize=None, use_rttm=True)
            cnt = 0
            for item in dataset:
                cnt += 1
                feat = item[0]
                token_len = item[3]
                assert feat.shape == golden_feat_shape
                assert torch.equal(token_len, torch.tensor(5))

                if cnt == 1:
                    assert torch.equal(feat, sample)
                else:
                    assert torch.equal(feat, masked_sample)

            assert cnt == num_samples

    @pytest.mark.unit
    def test_feature_with_rttm_to_text_bpe_dataset(self, test_data_dir):
        tokenizer_path = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
        tokenizer = tokenizers.AutoTokenizer(pretrained_model_name='bert-base-cased', vocab_file=tokenizer_path)
        num_samples = 2
        golden_feat_shape = (80, 10)
        sample = torch.ones(80, 10)
        masked_sample = sample * FeatureToCharDataset.ZERO_LEVEL_SPEC_DB_VAL
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest_input.json')
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                feat_file = os.path.join(tmpdir, f"feat_0.pt")
                torch.save(sample, feat_file)

                rttm_file = os.path.join(tmpdir, f"rttm_0.rttm")
                with open(rttm_file, "w") as fout:
                    fout.write(f"SPEAKER <NA> 1 0 1 <NA> <NA> speech <NA> <NA>\n")

                entry = {
                    'audio_filepath': "",
                    'feature_file': feat_file,
                    'rttm_file': rttm_file,
                    'duration': 100000,
                    "text": "a b c",
                }
                fp.write(json.dumps(entry) + '\n')

                # second sample where all frames are not masked
                feat_file = os.path.join(tmpdir, f"feat_1.pt")
                torch.save(sample, feat_file)

                rttm_file = os.path.join(tmpdir, f"rttm_1.rttm")
                with open(rttm_file, "w") as fout:
                    fout.write(f"SPEAKER <NA> 1 0 0 <NA> <NA> speech <NA> <NA>\n")

                entry = {
                    'audio_filepath': "",
                    'feature_file': feat_file,
                    'rttm_file': rttm_file,
                    'duration': 100000,
                    "text": "a b c",
                }
                fp.write(json.dumps(entry) + '\n')

            dataset = FeatureToBPEDataset(manifest_path, tokenizer=tokenizer, normalize=None, use_rttm=True)
            cnt = 0
            for item in dataset:
                cnt += 1
                feat = item[0]
                token_len = item[3]
                assert feat.shape == golden_feat_shape
                assert torch.equal(token_len, torch.tensor(5))

                if cnt == 1:
                    assert torch.equal(feat, sample)
                else:
                    assert torch.equal(feat, masked_sample)

            assert cnt == num_samples


class TestAudioDatasets:
    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 2])
    @pytest.mark.parametrize('num_targets', [1, 3])
    def test_list_to_multichannel(self, num_channels, num_targets):
        """Test conversion of a list of arrays into 
        """
        random_seed = 42
        num_samples = 1000

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Multi-channel signal
        golden_target = _rng.normal(size=(num_channels * num_targets, num_samples))

        # Create a list of num_targets signals with num_channels channels
        target_list = [golden_target[n * num_channels : (n + 1) * num_channels, :] for n in range(num_targets)]

        # Check the original signal is not modified
        assert (ASRAudioProcessor.list_to_multichannel(golden_target) == golden_target).all()
        # Check the list is converted back to the original signal
        assert (ASRAudioProcessor.list_to_multichannel(target_list) == golden_target).all()

    @pytest.mark.unit
    def test_audio_collate_fn(self):
        """Test `_audio_collate_fn`
        """
        batch_size = 16
        random_seed = 42
        atol = 1e-5

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        signal_to_channels = {
            'input_signal': 2,
            'target_signal': 1,
            'reference_signal': 1,
        }

        signal_to_length = {
            'input_signal': _rng.integers(low=5, high=25, size=batch_size),
            'target_signal': _rng.integers(low=5, high=25, size=batch_size),
            'reference_signal': _rng.integers(low=5, high=25, size=batch_size),
        }

        # Generate batch
        batch = []
        for n in range(batch_size):
            item = dict()
            for signal, num_channels in signal_to_channels.items():
                random_signal = _rng.normal(size=(num_channels, signal_to_length[signal][n]))
                random_signal = np.squeeze(random_signal)  # get rid of channel dimention for single-channel
                item[signal] = torch.tensor(random_signal)
            batch.append(item)

        # Run UUT
        batched = _audio_collate_fn(batch)

        batched_signals = {
            'input_signal': batched[0].cpu().detach().numpy(),
            'target_signal': batched[2].cpu().detach().numpy(),
            'reference_signal': batched[4].cpu().detach().numpy(),
        }

        batched_lengths = {
            'input_signal': batched[1].cpu().detach().numpy(),
            'target_signal': batched[3].cpu().detach().numpy(),
            'reference_signal': batched[5].cpu().detach().numpy(),
        }

        # Check outputs
        for signal, b_signal in batched_signals.items():
            for n in range(batch_size):
                # Check length
                uut_length = batched_lengths[signal][n]
                golden_length = signal_to_length[signal][n]
                assert (
                    uut_length == golden_length
                ), f'Example {n} signal {signal} length mismatch: batched ({uut_length}) != golden ({golden_length})'

                uut_signal = b_signal[n][:uut_length, ...]
                golden_signal = batch[n][signal][:uut_length, ...].cpu().detach().numpy()
                assert np.allclose(
                    uut_signal, golden_signal, atol=atol
                ), f'Example {n} signal {signal} value mismatch.'

    @pytest.mark.unit
    def test_audio_to_target_dataset(self):
        """Test AudioWithTargetDataset in different configurations.

        Test below cover the following:
        1) no constraints
        2) filtering based on signal duration
        3) use with channel selector
        4) use with fixed audio duration and random subsegments
        5) collate a batch of items

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'target_filepath': 'path/to/path_to_target.wav',
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
            'target_signal': 2,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        data_key = {
            'input_signal': 'input_filepath',
            'target_signal': 'target_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples[n]))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_duration_samples[n]))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:

            # Build metadata for manifest
            metadata = []

            for n in range(num_examples):

                meta = dict()

                for signal in data:
                    # filenames
                    signal_filename = f'{signal}_{n:02d}.wav'

                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')

                    # update metadata
                    meta[data_key[signal]] = signal_filename

                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                sample_rate=sample_rate,
            )

            # Also test the corresponding factory
            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': data_key['target_signal'],
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_dataset(config)

            # Test number of channels
            for signal in data:
                assert data_num_channels[signal] == dataset.num_channels(
                    signal
                ), f'Num channels not correct for signal {signal}'
                assert data_num_channels[signal] == dataset_factory.num_channels(
                    signal
                ), f'Num channels not correct for signal {signal}'

            # Test returned examples
            for n in range(num_examples):
                item = dataset.__getitem__(n)
                item_factory = dataset_factory.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for example {n}, signal {signal} (random seed {random_seed})'

                    item_factory_signal = item_factory[signal].cpu().detach().numpy()
                    assert np.allclose(
                        item_factory_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

            # Test 2
            # - Filtering based on signal duration
            min_duration = 3.5
            max_duration = 7.5

            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                min_duration=min_duration,
                max_duration=max_duration,
                sample_rate=sample_rate,
            )

            filtered_examples = [n for n, val in enumerate(data_duration) if min_duration <= val <= max_duration]

            for n in range(len(dataset)):
                item = dataset.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][filtered_examples[n]]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 2: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 3
            # - Use channel selector
            channel_selector = {
                'input_signal': [0, 2],
                'target_signal': 1,
            }

            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                input_channel_selector=channel_selector['input_signal'],
                target_channel_selector=channel_selector['target_signal'],
                sample_rate=sample_rate,
            )

            for n in range(len(dataset)):
                item = dataset.__getitem__(n)

                for signal in data:
                    cs = channel_selector[signal]
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n][cs, ...]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 3: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 4
            # - Use fixed duration (random segment selection)
            audio_duration = 4.0
            audio_duration_samples = int(np.floor(audio_duration * sample_rate))

            filtered_examples = [n for n, val in enumerate(data_duration) if val >= audio_duration]

            for random_offset in [True, False]:
                # Test subsegments with the default fixed offset and a random offset

                dataset = AudioToTargetDataset(
                    manifest_filepath=manifest_filepath,
                    input_key=data_key['input_signal'],
                    target_key=data_key['target_signal'],
                    sample_rate=sample_rate,
                    min_duration=audio_duration,
                    audio_duration=audio_duration,
                    random_offset=random_offset,  # random offset when selecting subsegment
                )

                for n in range(len(dataset)):
                    item = dataset.__getitem__(n)

                    golden_start = golden_end = None
                    for signal in data:
                        item_signal = item[signal].cpu().detach().numpy()
                        full_golden_signal = data[signal][filtered_examples[n]]

                        # Find random segment using correlation on the first channel
                        # of the first signal, and then use it fixed for other signals
                        if golden_start is None:
                            golden_start = get_segment_start(
                                signal=full_golden_signal[0, :], segment=item_signal[0, :]
                            )
                            if not random_offset:
                                assert (
                                    golden_start == 0
                                ), f'Expecting the signal to start at 0 when random_offset is False'

                            golden_end = golden_start + audio_duration_samples
                        golden_signal = full_golden_signal[..., golden_start:golden_end]

                        # Test length is correct
                        assert (
                            item_signal.shape[-1] == audio_duration_samples
                        ), f'Test 4: Signal length ({item_signal.shape[-1]}) not matching the expected length ({audio_duration_samples})'

                        assert (
                            item_signal.shape == golden_signal.shape
                        ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                        # Test signal values
                        assert np.allclose(
                            item_signal, golden_signal, atol=atol
                        ), f'Test 4: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 5:
            # - Test collate_fn
            batch_size = 16
            batch = [dataset.__getitem__(n) for n in range(batch_size)]
            batched = dataset.collate_fn(batch)

            for n, signal in enumerate(data.keys()):
                signal_shape = batched[2 * n].shape
                signal_len = batched[2 * n + 1]

                assert signal_shape == (
                    batch_size,
                    data_num_channels[signal],
                    audio_duration_samples,
                ), f'Test 5: Unexpected signal {signal} shape {signal_shape}'
                assert len(signal_len) == batch_size, f'Test 5: Unexpected length of signal_len ({len(signal_len)})'
                assert all(signal_len == audio_duration_samples), f'Test 5: Unexpected signal_len {signal_len}'

    @pytest.mark.unit
    def test_audio_to_target_dataset_with_target_list(self):
        """Test AudioWithTargetDataset when the input manifest has a list
        of audio files in the target key.

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'target_filepath': ['path/to/path_to_target_ch0.wav', 'path/to/path_to_target_ch1.wav'],
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
            'target_signal': 2,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        data_key = {
            'input_signal': 'input_filepath',
            'target_signal': 'target_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples[n]))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_duration_samples[n]))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:

            # Build metadata for manifest
            metadata = []

            for n in range(num_examples):

                meta = dict()

                for signal in data:
                    if signal == 'target_signal':
                        # Save targets as individual files
                        signal_filename = []
                        for ch in range(data_num_channels[signal]):
                            # add current filename
                            signal_filename.append(f'{signal}_{n:02d}_ch_{ch}.wav')
                            # write audio file
                            sf.write(
                                os.path.join(test_dir, signal_filename[-1]),
                                data[signal][n][ch, :],
                                sample_rate,
                                'float',
                            )
                    else:
                        # single file
                        signal_filename = f'{signal}_{n:02d}.wav'

                        # write audio files
                        sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')

                    # update metadata
                    meta[data_key[signal]] = signal_filename

                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                sample_rate=sample_rate,
            )

            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': data_key['target_signal'],
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_dataset(config)

            for n in range(num_examples):
                item = dataset.__getitem__(n)
                item_factory = dataset_factory.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for example {n}, signal {signal} (random seed {random_seed})'

                    item_factory_signal = item_factory[signal].cpu().detach().numpy()
                    assert np.allclose(
                        item_factory_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

            # Test 2
            # Set target as the first channel of input_filepath and all files listed in target_filepath.
            # In this case, the target will have 3 channels.
            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=[data_key['input_signal'], data_key['target_signal']],
                target_channel_selector=0,
                sample_rate=sample_rate,
            )

            for n in range(num_examples):
                item = dataset.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    if signal == 'target_signal':
                        # add the first channel of the input
                        golden_signal = np.concatenate([data['input_signal'][n][0:1, ...], golden_signal], axis=0)
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 2: Failed for example {n}, signal {signal} (random seed {random_seed})'

    @pytest.mark.unit
    def test_audio_to_target_dataset_for_inference(self):
        """Test AudioWithTargetDataset when target_key is
        not set, i.e., it is `None`. This is the case, e.g., when
        running inference, and a target is not available.

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        data_key = {
            'input_signal': 'input_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples[n]))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_duration_samples[n]))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:
            # Build metadata for manifest
            metadata = []
            for n in range(num_examples):
                meta = dict()
                for signal in data:
                    # filenames
                    signal_filename = f'{signal}_{n:02d}.wav'
                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')
                    # update metadata
                    meta[data_key[signal]] = signal_filename
                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            dataset = AudioToTargetDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=None,  # target_signal will be empty
                sample_rate=sample_rate,
            )

            # Also test the corresponding factory
            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': None,
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_dataset(config)

            for n in range(num_examples):
                item = dataset.__getitem__(n)
                item_factory = dataset_factory.__getitem__(n)

                # Check target is None
                assert item['target_signal'].numel() == 0, 'target_signal is expected to be empty.'
                assert item_factory['target_signal'].numel() == 0, 'target_signal is expected to be empty.'

                # Check valid signals
                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for example {n}, signal {signal} (random seed {random_seed})'

                    item_factory_signal = item_factory[signal].cpu().detach().numpy()
                    assert np.allclose(
                        item_factory_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

    @pytest.mark.unit
    def test_audio_to_target_with_reference_dataset(self):
        """Test AudioWithTargetWithReferenceDataset in different configurations.

        1) reference synchronized with input and target
        2) reference not synchronized

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'target_filepath': 'path/to/path_to_target.wav',
            'reference_filepath': 'path/to/path_to_reference.wav',
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
            'target_signal': 2,
            'reference_signal': 1,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        data_key = {
            'input_signal': 'input_filepath',
            'target_signal': 'target_filepath',
            'reference_signal': 'reference_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples[n]))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_duration_samples[n]))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:

            # Build metadata for manifest
            metadata = []

            for n in range(num_examples):

                meta = dict()

                for signal in data:
                    # filenames
                    signal_filename = f'{signal}_{n:02d}.wav'

                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')

                    # update metadata
                    meta[data_key[signal]] = signal_filename

                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            # - Reference is not synchronized with input and target, so whole reference signal will be loaded
            dataset = AudioToTargetWithReferenceDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                reference_key=data_key['reference_signal'],
                reference_is_synchronized=False,
                sample_rate=sample_rate,
            )

            # Also test the corresponding factory
            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': data_key['target_signal'],
                'reference_key': data_key['reference_signal'],
                'reference_is_synchronized': False,
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_with_reference_dataset(config)

            for n in range(num_examples):
                item = dataset.__getitem__(n)
                item_factory = dataset_factory.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for example {n}, signal {signal} (random seed {random_seed})'

                    item_factory_signal = item_factory[signal].cpu().detach().numpy()
                    assert np.allclose(
                        item_factory_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

            # Test 2
            # - Use fixed duration (random segment selection)
            # - Reference is synchronized with input and target, so the same segment of reference signal will be loaded
            audio_duration = 4.0
            audio_duration_samples = int(np.floor(audio_duration * sample_rate))
            dataset = AudioToTargetWithReferenceDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                reference_key=data_key['reference_signal'],
                reference_is_synchronized=True,
                sample_rate=sample_rate,
                min_duration=audio_duration,
                audio_duration=audio_duration,
                random_offset=True,
            )

            filtered_examples = [n for n, val in enumerate(data_duration) if val >= audio_duration]

            for n in range(len(dataset)):
                item = dataset.__getitem__(n)

                golden_start = golden_end = None
                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    full_golden_signal = data[signal][filtered_examples[n]]

                    # Find random segment using correlation on the first channel
                    # of the first signal, and then use it fixed for other signals
                    if golden_start is None:
                        golden_start = get_segment_start(signal=full_golden_signal[0, :], segment=item_signal[0, :])
                        golden_end = golden_start + audio_duration_samples
                    golden_signal = full_golden_signal[..., golden_start:golden_end]

                    # Test length is correct
                    assert (
                        item_signal.shape[-1] == audio_duration_samples
                    ), f'Test 2: Signal {signal} length ({item_signal.shape[-1]}) not matching the expected length ({audio_duration_samples})'

                    # Test signal values
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 2: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 3
            # - Use fixed duration (random segment selection)
            # - Reference is not synchronized with input and target, so whole reference signal will be loaded
            audio_duration = 4.0
            audio_duration_samples = int(np.floor(audio_duration * sample_rate))
            dataset = AudioToTargetWithReferenceDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                reference_key=data_key['reference_signal'],
                reference_is_synchronized=False,
                sample_rate=sample_rate,
                min_duration=audio_duration,
                audio_duration=audio_duration,
                random_offset=True,
            )

            filtered_examples = [n for n, val in enumerate(data_duration) if val >= audio_duration]

            for n in range(len(dataset)):
                item = dataset.__getitem__(n)

                golden_start = golden_end = None
                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    full_golden_signal = data[signal][filtered_examples[n]]

                    if signal == 'reference_signal':
                        # Complete signal is loaded for reference
                        golden_signal = full_golden_signal
                    else:
                        # Find random segment using correlation on the first channel
                        # of the first signal, and then use it fixed for other signals
                        if golden_start is None:
                            golden_start = get_segment_start(
                                signal=full_golden_signal[0, :], segment=item_signal[0, :]
                            )
                            golden_end = golden_start + audio_duration_samples
                        golden_signal = full_golden_signal[..., golden_start:golden_end]

                        # Test length is correct
                        assert (
                            item_signal.shape[-1] == audio_duration_samples
                        ), f'Test 3: Signal {signal} length ({item_signal.shape[-1]}) not matching the expected length ({audio_duration_samples})'
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    # Test signal values
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 3: Failed for example {n}, signal {signal} (random seed {random_seed})'

            # Test 4:
            # - Test collate_fn
            batch_size = 16
            batch = [dataset.__getitem__(n) for n in range(batch_size)]
            _ = dataset.collate_fn(batch)

    @pytest.mark.unit
    def test_audio_to_target_with_embedding_dataset(self):
        """Test AudioWithTargetWithEmbeddingDataset.

        In this use case, each line of the manifest file has the following format:
        ```
        {
            'input_filepath': 'path/to/input.wav',
            'target_filepath': 'path/to/path_to_target.wav',
            'embedding_filepath': 'path/to/path_to_embedding.npy',
            'duration': duration_of_input,
        }
        ```
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 25
        data_num_channels = {
            'input_signal': 4,
            'target_signal': 2,
            'embedding_vector': 1,
        }
        data_min_duration = 2.0
        data_max_duration = 8.0
        embedding_length = 64  # 64-dimensional embedding vector
        data_key = {
            'input_signal': 'input_filepath',
            'target_signal': 'target_filepath',
            'embedding_vector': 'embedding_filepath',
        }

        # Tolerance
        atol = 1e-6

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
        data_duration_samples = np.floor(data_duration * sample_rate).astype(int)

        data = dict()
        for signal, num_channels in data_num_channels.items():
            data[signal] = []
            for n in range(num_examples):
                data_length = embedding_length if signal == 'embedding_vector' else data_duration_samples[n]

                if num_channels == 1:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(data_length))
                else:
                    random_signal = _rng.uniform(low=-0.5, high=0.5, size=(num_channels, data_length))
                data[signal].append(random_signal)

        with tempfile.TemporaryDirectory() as test_dir:

            # Build metadata for manifest
            metadata = []

            for n in range(num_examples):

                meta = dict()

                for signal in data:
                    if signal == 'embedding_vector':
                        signal_filename = f'{signal}_{n:02d}.npy'
                        np.save(os.path.join(test_dir, signal_filename), data[signal][n])

                    else:
                        # filenames
                        signal_filename = f'{signal}_{n:02d}.wav'

                        # write audio files
                        sf.write(os.path.join(test_dir, signal_filename), data[signal][n].T, sample_rate, 'float')

                    # update metadata
                    meta[data_key[signal]] = signal_filename

                meta['duration'] = data_duration[n]
                metadata.append(meta)

            # Save manifest
            manifest_filepath = os.path.join(test_dir, 'manifest.json')
            write_manifest(manifest_filepath, metadata)

            # Test 1
            # - No constraints on channels or duration
            dataset = AudioToTargetWithEmbeddingDataset(
                manifest_filepath=manifest_filepath,
                input_key=data_key['input_signal'],
                target_key=data_key['target_signal'],
                embedding_key=data_key['embedding_vector'],
                sample_rate=sample_rate,
            )

            # Also test the corresponding factory
            config = {
                'manifest_filepath': manifest_filepath,
                'input_key': data_key['input_signal'],
                'target_key': data_key['target_signal'],
                'embedding_key': data_key['embedding_vector'],
                'sample_rate': sample_rate,
            }
            dataset_factory = audio_to_audio_dataset.get_audio_to_target_with_embedding_dataset(config)

            for n in range(num_examples):
                item = dataset.__getitem__(n)
                item_factory = dataset_factory.__getitem__(n)

                for signal in data:
                    item_signal = item[signal].cpu().detach().numpy()
                    golden_signal = data[signal][n]
                    assert (
                        item_signal.shape == golden_signal.shape
                    ), f'Signal {signal}: item shape {item_signal.shape} not matching reference shape {golden_signal.shape}'
                    assert np.allclose(
                        item_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for example {n}, signal {signal} (random seed {random_seed})'

                    item_factory_signal = item_factory[signal].cpu().detach().numpy()
                    assert np.allclose(
                        item_factory_signal, golden_signal, atol=atol
                    ), f'Test 1: Failed for factory example {n}, signal {signal} (random seed {random_seed})'

            # Test 2:
            # - Test collate_fn
            batch_size = 16
            batch = [dataset.__getitem__(n) for n in range(batch_size)]
            _ = dataset.collate_fn(batch)


class TestUtilityFunctions:
    @pytest.mark.unit
    @pytest.mark.parametrize('cache_audio', [False, True])
    def test_cache_datastore_manifests(self, cache_audio: bool):
        """Test caching of manifest and audio files.
        """
        # Data setup
        random_seed = 42
        sample_rate = 16000
        num_examples = 10
        num_manifests = 2
        data_duration = 1.0

        # Generate random signals
        _rng = np.random.default_rng(seed=random_seed)

        # Input and target signals have the same duration
        data_duration_samples = int(data_duration * sample_rate)

        with tempfile.TemporaryDirectory() as test_dir:
            test_store_dir = os.path.join(test_dir, 'store')
            os.mkdir(test_store_dir)

            # Prepare metadata and audio files
            manifest_filepaths = []
            audio_files = []
            for m in range(num_manifests):
                manifest_dir = os.path.join(test_store_dir, f'manifest_{m}')
                os.mkdir(manifest_dir)
                manifest_filepath = os.path.join(manifest_dir, 'manifest.json')

                metadata = []
                data = _rng.uniform(low=-0.5, high=0.5, size=(data_duration_samples, num_examples))
                for n in range(num_examples):
                    audio_filepath = f'manifest_{m}_audio_{n:02d}.wav'
                    audio_file = os.path.join(manifest_dir, audio_filepath)
                    # Write audio file
                    sf.write(audio_file, data[:, n], sample_rate, 'float')
                    # Update metadata
                    metadata.append(
                        {
                            'audio_filepath': audio_filepath,
                            'duration': data_duration,
                            'text': f'text for example {n:02d}',
                        }
                    )
                    # Update audio files
                    audio_files.append(audio_file)

                # Save manifest
                write_manifest(manifest_filepath, metadata)
                manifest_filepaths.append(manifest_filepath)

            # Cache location
            test_cache_dir = os.path.join(test_dir, 'cache')

            # Instead of using AIS, copy object from store dir to cache dir
            def fake_get(self):
                # Object path relative to store path
                object_path = os.path.relpath(self.store_path, start=test_store_dir)
                # Copy to fake local path
                self._local_path = os.path.join(test_cache_dir, object_path)
                os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
                shutil.copy(self.store_path, self.local_path)
                # Return path as in the original get
                return self.local_path

            with mock.patch(
                'nemo.collections.asr.data.audio_to_text.is_datastore_path', lambda x: True
            ), mock.patch.object(DataStoreObject, 'get', fake_get):
                # Use a single worker for this test to avoid failure with mock & multiprocessing (#5607)
                cache_datastore_manifests(manifest_filepaths, cache_audio=cache_audio, num_workers=1)

            # Manifests need to be compared
            store_files_to_compare = manifest_filepaths
            if cache_audio:
                # Audio needs to be compared
                store_files_to_compare += audio_files

            # Compare files
            for f_store in store_files_to_compare:
                f_cache = os.path.join(test_cache_dir, os.path.relpath(f_store, test_store_dir))
                assert filecmp.cmp(f_store, f_cache, shallow=False), f'Files {f_store} and {f_cache} do not match.'
