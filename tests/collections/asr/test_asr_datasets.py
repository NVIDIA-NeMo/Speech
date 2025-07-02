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

from nemo.collections.asr.data import audio_to_text_dataset
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
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.common import tokenizers
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
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
    def test_tarred_dataset_filter(self, test_data_dir):
        """
        Checks for
            1. file count when manifest len is less than tarred dataset
            2. Ignoring files in manifest that are not in tarred balls

        """
        manifest_path = os.path.abspath(
            os.path.join(test_data_dir, 'asr/tarred_an4/tarred_duplicate_audio_manifest.json')
        )

        # Test braceexpand loading
        tarpath = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/audio_{0..1}.tar'))
        ds_braceexpand = TarredAudioToCharDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, sample_rate=16000
        )
        assert len(ds_braceexpand) == 6
        count = 0
        for _ in ds_braceexpand:
            count += 1
        assert count == 5  # file ending with sub is not part of tar ball

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

    @pytest.mark.xfail(
        reason="DALI ASR Dataset's preprocessor is not patched with padding inconsistency fix (PR #13827)"
    )
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
            ref_dataset = audio_to_text_dataset.get_char_dataset(
                config=dataset_cfg,
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


class TestUtilityFunctions:
    @pytest.mark.unit
    @pytest.mark.parametrize('cache_audio', [False, True])
    def test_cache_datastore_manifests(self, cache_audio: bool):
        """Test caching of manifest and audio files."""
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

            with (
                mock.patch('nemo.collections.asr.data.audio_to_text.is_datastore_path', lambda x: True),
                mock.patch.object(DataStoreObject, 'get', fake_get),
            ):
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
