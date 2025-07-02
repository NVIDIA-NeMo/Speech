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

import pytest
import torch.cuda

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, WaveformFeaturizer
from nemo.collections.asr.parts.utils.speaker_utils import get_vad_out_from_rttm_line, read_rttm_lines


def is_rttm_length_too_long(rttm_file_path, wav_len_in_sec):
    """
    Check if the maximum RTTM duration exceeds the length of the provided audio file.

    Args:
        rttm_file_path (str): Path to the RTTM file.
        wav_len_in_sec (float): Length of the audio file in seconds.

    Returns:
        bool: True if the maximum RTTM duration is less than or equal to the length of the audio file, False otherwise.
    """
    rttm_lines = read_rttm_lines(rttm_file_path)
    max_rttm_sec = 0
    for line in rttm_lines:
        start, dur = get_vad_out_from_rttm_line(line)
        max_rttm_sec = max(max_rttm_sec, start + dur)
    return max_rttm_sec <= wav_len_in_sec


class TestAudioToSpeechE2ESpkDiarDataset:

    @pytest.mark.unit
    def test_e2e_speaker_diar_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/diarizer/lsm_val.json'))

        batch_size = 4
        num_samples = 8
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)
            fb_featurizer = FilterbankFeatures(
                sample_rate=featurizer.sample_rate,
                n_window_size=int(0.025 * featurizer.sample_rate),
                n_window_stride=int(0.01 * featurizer.sample_rate),
                dither=False,
            )

            dataset = AudioToSpeechE2ESpkDiarDataset(
                manifest_filepath=f.name,
                soft_label_thres=0.5,
                session_len_sec=90,
                num_spks=4,
                featurizer=featurizer,
                window_stride=0.01,
                global_rank=0,
                soft_targets=False,
                device=device,
                fb_featurizer=fb_featurizer,
            )
            dataloader_instance = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=dataset.eesd_train_collate_fn,
                drop_last=False,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            assert len(dataloader_instance) == (num_samples / batch_size)  # Check if the number of batches is correct
            batch_counts = len(dataloader_instance)

            deviation_thres_rate = 0.01  # 1% deviation allowed
            for batch_index, batch in enumerate(dataloader_instance):
                if batch_index != batch_counts - 1:
                    assert len(batch) == batch_size, "Batch size does not match the expected value"
                audio_signals, audio_signal_len, targets, target_lens = batch
                for sample_index in range(audio_signals.shape[0]):
                    dataloader_audio_in_sec = audio_signal_len[sample_index].item()
                    data_dur_in_sec = abs(
                        data_dict_list[batch_size * batch_index + sample_index]['duration'] * featurizer.sample_rate
                        - dataloader_audio_in_sec
                    )
                    assert (
                        data_dur_in_sec <= deviation_thres_rate * dataloader_audio_in_sec
                    ), "Duration deviation exceeds 1%"
                assert not torch.isnan(audio_signals).any(), "audio_signals tensor contains NaN values"
                assert not torch.isnan(audio_signal_len).any(), "audio_signal_len tensor contains NaN values"
                assert not torch.isnan(targets).any(), "targets tensor contains NaN values"
                assert not torch.isnan(target_lens).any(), "target_lens tensor contains NaN values"
