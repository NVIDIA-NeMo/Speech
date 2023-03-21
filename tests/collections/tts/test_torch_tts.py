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
from pathlib import Path

import pytest
import torch

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishPhonemesTokenizer
from nemo.collections.tts.data.dataset import TTSDataset
from nemo.collections.tts.g2p.models.en_us_arpabet import EnglishG2p
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir


class TestTTSDataset:
    @pytest.mark.unit
    @pytest.mark.run_only_on('CPU')
    def test_dataset(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')

        dataset = TTSDataset(
            manifest_filepath=manifest_path,
            sample_rate=22050,
            sup_data_types=["pitch"],
            sup_data_path=sup_path,
            text_tokenizer=EnglishPhonemesTokenizer(
                punct=True,
                stresses=True,
                chars=True,
                space=' ',
                apostrophe=True,
                pad_with_space=True,
                g2p=EnglishG2p(),
            ),
        )

        dataloader = torch.utils.data.DataLoader(dataset, 2, collate_fn=dataset._collate_fn)
        data, _, _, _, _, _ = next(iter(dataloader))

    @pytest.mark.unit
    @pytest.mark.run_only_on('CPU')
    def test_raise_exception_on_not_supported_sup_data_types(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')
        with pytest.raises(NotImplementedError):
            dataset = TTSDataset(
                manifest_filepath=manifest_path,
                sample_rate=22050,
                sup_data_types=["not_supported_sup_data_type"],
                sup_data_path=sup_path,
                text_tokenizer=EnglishPhonemesTokenizer(
                    punct=True,
                    stresses=True,
                    chars=True,
                    space=' ',
                    apostrophe=True,
                    pad_with_space=True,
                    g2p=EnglishG2p(),
                ),
            )

    @pytest.mark.unit
    @pytest.mark.run_only_on('CPU')
    def test_raise_exception_on_not_supported_window(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')
        with pytest.raises(NotImplementedError):
            dataset = TTSDataset(
                manifest_filepath=manifest_path,
                sample_rate=22050,
                sup_data_types=["pitch"],
                sup_data_path=sup_path,
                window="not_supported_window",
                text_tokenizer=EnglishPhonemesTokenizer(
                    punct=True,
                    stresses=True,
                    chars=True,
                    space=' ',
                    apostrophe=True,
                    pad_with_space=True,
                    g2p=EnglishG2p(),
                ),
            )

    @pytest.mark.unit
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.parametrize("sup_data_type", ["voiced_mask", "p_voiced"])
    def test_raise_exception_on_missing_pitch_sup_data_type_if_use_voiced(self, test_data_dir, sup_data_type):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')
        with pytest.raises(ValueError):
            dataset = TTSDataset(
                manifest_filepath=manifest_path,
                sample_rate=22050,
                sup_data_types=[sup_data_type],
                sup_data_path=sup_path,
                window="hann",
                text_tokenizer=EnglishPhonemesTokenizer(
                    punct=True,
                    stresses=True,
                    chars=True,
                    space=' ',
                    apostrophe=True,
                    pad_with_space=True,
                    g2p=EnglishG2p(),
                ),
            )

    @pytest.mark.unit
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.parametrize(
        "sup_data_types, output_indices",
        [
            (["p_voiced", "pitch", "voiced_mask"], [-4, -3, -1]),
            (["voiced_mask", "pitch"], [-3, -2]),
            (["pitch", "p_voiced"], [-3, -1]),
            (["pitch"], [-2]),
        ],
    )
    def test_save_voiced_items_if_pt_file_not_exist(self, test_data_dir, sup_data_types, output_indices, tmp_path):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = tmp_path / "sup_data"
        print(f"sup_path={sup_path}")
        dataset = TTSDataset(
            manifest_filepath=manifest_path,
            sample_rate=22050,
            sup_data_types=sup_data_types,
            sup_data_path=sup_path,
            text_tokenizer=EnglishPhonemesTokenizer(
                punct=True,
                stresses=True,
                chars=True,
                space=' ',
                apostrophe=True,
                pad_with_space=True,
                g2p=EnglishG2p(),
            ),
        )

        # load manifest
        audio_filepaths = []
        with open(manifest_path, 'r', encoding="utf-8") as fjson:
            for line in fjson:
                audio_filepaths.append(json.loads(line)["audio_filepath"])
        base_data_dir = get_base_dir(audio_filepaths)

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, collate_fn=dataset._collate_fn)
        for batch, audio_filepath in zip(dataloader, audio_filepaths):
            rel_audio_path = Path(audio_filepath).relative_to(base_data_dir).with_suffix("")
            rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")

            for sup_data_type, output_index in zip(sup_data_types, output_indices):
                sup_data = batch[output_index]
                sup_data = sup_data.squeeze(0)
                assert sup_data is not None
                assert torch.equal(sup_data, torch.load(f"{sup_path}/{sup_data_type}/{rel_audio_path_as_text_id}.pt"))

                if sup_data_type == "pitch":
                    pitch_lengths = batch[output_index + 1]
                    pitch_lengths = pitch_lengths.squeeze(0)
                    assert pitch_lengths is not None

            # test pitch, voiced_mask, and p_voiced do not have the same values.
            if len(sup_data_types) == 3:
                x = torch.load(f"{sup_path}/{sup_data_types[0]}/{rel_audio_path_as_text_id}.pt")
                y = torch.load(f"{sup_path}/{sup_data_types[1]}/{rel_audio_path_as_text_id}.pt")
                z = torch.load(f"{sup_path}/{sup_data_types[2]}/{rel_audio_path_as_text_id}.pt")
                assert not torch.equal(x, y)
                assert not torch.equal(x, z)
