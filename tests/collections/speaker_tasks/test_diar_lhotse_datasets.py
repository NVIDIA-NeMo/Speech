# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from unittest import mock

import pytest
import torch
import torch.cuda
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_diar_label_lhotse import LhotseAudioToSpeechE2ESpkDiarDataset
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config


def get_train_ds_config(manifest_filepath, batch_size, num_workers) -> DictConfig:
    return DictConfig(
        {
            'manifest_filepath': manifest_filepath,
            'sample_rate': 16000,
            'num_spks': 4,
            'session_len_sec': 90,
            'soft_label_thres': 0.5,
            'soft_targets': False,
            'labels': None,
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'validation_mode': False,
            'use_lhotse': True,
            'use_bucketing': True,
            'num_buckets': 10,
            'bucket_duration_bins': [10, 20, 30, 40, 50, 60, 70, 80, 90],
            'pin_memory': True,
            'min_duration': 80,
            'max_duration': 90,
            'batch_duration': 400,
            'quadratic_duration': 1200,
            'bucket_buffer_size': 20000,
            'shuffle_buffer_size': 10000,
            'window_stride': 0.01,
            'subsampling_factor': 8,
        }
    )


def get_validation_ds_config(manifest_filepath, batch_size, num_workers) -> DictConfig:
    return DictConfig(
        {
            'manifest_filepath': manifest_filepath,
            'is_tarred': False,
            'tarred_audio_filepaths': None,
            'sample_rate': 16000,
            'num_spks': 4,
            'session_len_sec': 90,
            'soft_label_thres': 0.5,
            'soft_targets': False,
            'labels': None,
            'batch_size': batch_size,
            'shuffle': False,
            'seq_eval_mode': True,
            'num_workers': num_workers,
            'validation_mode': True,
            'use_lhotse': False,
            'use_bucketing': False,
            'drop_last': False,
            'pin_memory': True,
            'window_stride': 0.01,
            'subsampling_factor': 8,
        }
    )


def get_test_ds_config(manifest_filepath, batch_size, num_workers) -> DictConfig:
    return DictConfig(
        {
            'manifest_filepath': manifest_filepath,
            'is_tarred': False,
            'tarred_audio_filepaths': None,
            'sample_rate': 16000,
            'num_spks': 4,
            'session_len_sec': 90,
            'soft_label_thres': 0.5,
            'soft_targets': False,
            'labels': None,
            'batch_size': batch_size,
            'shuffle': False,
            'seq_eval_mode': True,
            'num_workers': num_workers,
            'validation_mode': True,
            'use_lhotse': False,
            'use_bucketing': False,
            'drop_last': False,
            'pin_memory': True,
            'window_stride': 0.01,
            'subsampling_factor': 8,
        }
    )


class TestLhotseAudioToSpeechE2ESpkDiarDataset:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, num_workers, split",
        [
            (4, 8, 'train'),  # Example 1
            (4, 0, 'train'),  # Example 2
            (2, 4, 'validation'),  # Example 3
            (8, 2, 'test'),  # Example 4
        ],
    )
    def test_e2e_speaker_diar_lhotse_dataset(self, test_data_dir, batch_size, num_workers, split):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/diarizer/lsm_val.json'))
        num_samples = 8
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        data_dict_list = []
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r', encoding='utf-8') as mfile:
                for ix, line in enumerate(mfile):
                    if ix >= num_samples:
                        break

                    line = line.replace("tests/data/", test_data_dir + "/").replace("\n", "")
                    f.write(f"{line}\n")
                    data_dict = json.loads(line)
                    data_dict_list.append(data_dict)

            f.seek(0)
            config = None
            if split == 'train':
                config = get_train_ds_config(manifest_filepath=f.name, batch_size=batch_size, num_workers=num_workers)
            elif split == 'validation':
                config = get_train_ds_config(manifest_filepath=f.name, batch_size=batch_size, num_workers=num_workers)
            elif split == 'test':
                config = get_test_ds_config(manifest_filepath=f.name, batch_size=batch_size, num_workers=num_workers)

            dataloader_instance = get_lhotse_dataloader_from_config(
                config,
                global_rank=0,
                world_size=1,
                dataset=LhotseAudioToSpeechE2ESpkDiarDataset(cfg=config),
            )

            deviation_thres_rate = 0.01  # 1% deviation allowed
            for batch_index, batch in enumerate(dataloader_instance):
                audio_signals, audio_signal_len, targets, target_lens = batch
                for sample_index in range(audio_signals.shape[0]):
                    dataloader_audio_in_sec = audio_signal_len[sample_index].item()
                    data_dur_in_sec = abs(
                        data_dict_list[batch_size * batch_index + sample_index]['duration'] * config.sample_rate
                        - dataloader_audio_in_sec
                    )
                    assert (
                        data_dur_in_sec <= deviation_thres_rate * dataloader_audio_in_sec
                    ), "Duration deviation exceeds 1%"
                assert not torch.isnan(audio_signals).any(), "audio_signals tensor contains NaN values"
                assert not torch.isnan(audio_signal_len).any(), "audio_signal_len tensor contains NaN values"
                assert not torch.isnan(targets).any(), "targets tensor contains NaN values"
                assert not torch.isnan(target_lens).any(), "target_lens tensor contains NaN values"
