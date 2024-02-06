# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.mixins import TranscribeConfig, TranscriptionMixin
from nemo.collections.asr.parts.mixins.transcription import GenericTranscriptionType
from nemo.collections.asr.parts.utils import Hypothesis


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)

        self.execution_count = 0
        self.flag_begin = False
        self.flag_end = False

    def forward(self, x):
        # Input: [1, 1] Output = [1, 1
        out = self.encoder(x)
        return out


class TranscribableDummy(DummyModel, TranscriptionMixin):
    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio, trcfg)
        self.flag_begin = True

    def _transcribe_input_manifest_processing(self, audio_files: List[str], temp_dir: str, trcfg: TranscribeConfig):
        # Create a dummy manifest
        manifest_path = os.path.join(temp_dir, 'dummy_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as fp:
            for audio_file in audio_files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                fp.write(json.dumps(entry) + '\n')

        ds_config = {
            'paths2audio_files': audio_files,
            'batch_size': trcfg.batch_size,
            'temp_dir': temp_dir,
            'num_workers': trcfg.num_workers,
            'channel_selector': trcfg.channel_selector,
        }

        return ds_config

    def _setup_transcribe_dataloader(self, config: Dict) -> DataLoader:
        class DummyDataset(Dataset):
            def __init__(self, audio_files: List[str], config: Dict):
                self.audio_files = audio_files
                self.config = config

            def __getitem__(self, index):
                data = self.audio_files[index]
                data = torch.tensor([float(data)]).view(1)
                return data

            def __len__(self):
                return len(self.audio_files)

        dataset = DummyDataset(config['paths2audio_files'], config)

        return DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=False,
            drop_last=False,
        )

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        output = self(batch)
        return output

    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig) -> GenericTranscriptionType:
        self.execution_count += 1

        result = []
        for output in outputs:
            result.append(float(output.item()))

        if hasattr(trcfg, 'output_type') and trcfg.output_type == 'dict':
            results = {'output': result}
            return results

        if hasattr(trcfg, 'output_type') and trcfg.output_type == 'dict2':
            results = [{'output': res} for res in result]
            return results

        if hasattr(trcfg, 'output_type') and trcfg.output_type == 'tuple':
            result = tuple(result)
            return result

        # Pass list of results by default
        return result

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg)
        self.flag_end = True


@pytest.fixture()
def dummy_model():
    return TranscribableDummy()


class TestTranscriptionMixin:
    @pytest.mark.unit
    def test_constructor_non_instance(self):
        model = DummyModel()
        assert not isinstance(model, TranscriptionMixin)
        assert not hasattr(model, 'transcribe')

    @pytest.mark.unit
    def test_transcribe(self, dummy_model):
        dummy_model = dummy_model.eval()
        dummy_model.encoder.weight.data.fill_(1.0)
        dummy_model.encoder.bias.data.fill_(0.0)

        audio = ['1.0', '2.0', '3.0']
        outputs = dummy_model.transcribe(audio, batch_size=1)
        assert len(outputs) == 3
        assert outputs[0] == 1.0
        assert outputs[1] == 2.0
        assert outputs[2] == 3.0

    @pytest.mark.unit
    def test_transcribe_generator(self, dummy_model):
        dummy_model = dummy_model.eval()
        dummy_model.encoder.weight.data.fill_(1.0)
        dummy_model.encoder.bias.data.fill_(0.0)

        audio = ['1.0', '2.0', '3.0']

        transribe_config = TranscribeConfig(batch_size=1)
        generator = dummy_model.transcribe_generator(audio, override_config=transribe_config)

        outputs = []
        index = 1
        for result in generator:
            outputs.extend(result)
            assert len(result) == 1
            assert len(outputs) == index
            index += 1

        assert len(outputs) == 3
        assert outputs[0] == 1.0
        assert outputs[1] == 2.0
        assert outputs[2] == 3.0

    @pytest.mark.unit
    def test_transcribe_generator_explicit_stop_check(self, dummy_model):
        dummy_model = dummy_model.eval()
        dummy_model.encoder.weight.data.fill_(1.0)
        dummy_model.encoder.bias.data.fill_(0.0)

        audio = ['1.0', '2.0', '3.0']

        transribe_config = TranscribeConfig(batch_size=1)
        generator = dummy_model.transcribe_generator(audio, override_config=transribe_config)

        outputs = []
        index = 1
        while True:
            try:
                result = next(generator)
            except StopIteration:
                break
            outputs.extend(result)
            assert len(result) == 1
            assert len(outputs) == index
            index += 1

        assert len(outputs) == 3
        assert outputs[0] == 1.0
        assert outputs[1] == 2.0
        assert outputs[2] == 3.0

    @pytest.mark.unit
    def test_transcribe_check_flags(self, dummy_model):
        dummy_model = dummy_model.eval()

        audio = ['1.0', '2.0', '3.0']
        dummy_model.transcribe(audio, batch_size=1)
        assert dummy_model.flag_begin
        assert dummy_model.flag_end

    @pytest.mark.unit
    def test_transribe_override_config_incorrect(self, dummy_model):
        # Not subclassing TranscribeConfig
        @dataclass
        class OverrideConfig:
            batch_size: int = 1
            output_type: str = 'dict'

        dummy_model = dummy_model.eval()

        audio = [1.0, 2.0, 3.0]
        override_cfg = OverrideConfig(batch_size=1, output_type='dict')
        with pytest.raises(ValueError):
            _ = dummy_model.transcribe(audio, override_config=override_cfg)

    @pytest.mark.unit
    def test_transribe_override_config_correct(self, dummy_model):
        @dataclass
        class OverrideConfig(TranscribeConfig):
            output_type: str = 'dict'
            verbose: bool = False

        dummy_model = dummy_model.eval()
        dummy_model.encoder.weight.data.fill_(1.0)
        dummy_model.encoder.bias.data.fill_(0.0)

        audio = ['1.0', '2.0', '3.0']
        override_cfg = OverrideConfig(batch_size=1, output_type='dict')
        outputs = dummy_model.transcribe(audio, override_config=override_cfg)

        assert isinstance(outputs, dict)
        assert len(outputs) == 1
        assert dummy_model.execution_count == 3
        assert outputs['output'][0] == 1.0
        assert outputs['output'][1] == 2.0
        assert outputs['output'][2] == 3.0

        # Reset execution count
        dummy_model.execution_count = 0

        override_cfg = OverrideConfig(batch_size=1, output_type='dict2')
        outputs = dummy_model.transcribe(audio, override_config=override_cfg)

        # Output now is list of dict of value each
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert dummy_model.execution_count == 3
        assert outputs[0]['output'] == 1.0
        assert outputs[1]['output'] == 2.0
        assert outputs[2]['output'] == 3.0

        # Reset execution count
        dummy_model.execution_count = 0

        # Test tuple
        override_cfg = OverrideConfig(batch_size=1, output_type='tuple')
        outputs = dummy_model.transcribe(audio, override_config=override_cfg)

        assert isinstance(outputs, tuple)
        assert len(outputs) == 1
        assert dummy_model.execution_count == 3
        assert outputs[0][0] == 1.0
        assert outputs[0][1] == 2.0
        assert outputs[0][2] == 3.0

    pytest.mark.with_downloads()

    @pytest.mark.unit
    def test_transcribe_return_hypothesis(self, test_data_dir):
        model = ASRModel.from_pretrained("stt_en_conformer_ctc_small")
        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")

        # Numpy array test
        outputs = model.transcribe(audio_file, batch_size=1, return_hypotheses=True)
        assert len(outputs) == 1
        assert isinstance(outputs[0], Hypothesis)

        hyp = outputs[0]
        assert isinstance(hyp.text, str)
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert isinstance(hyp.alignments, torch.Tensor)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_transcribe_tensor(self, test_data_dir):
        model = ASRModel.from_pretrained("stt_en_conformer_ctc_small")

        # Load audio file
        import soundfile as sf

        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")
        audio, sr = sf.read(audio_file, dtype='float32')

        # Numpy array test
        outputs = model.transcribe(audio, batch_size=1)
        assert len(outputs) == 1
        assert isinstance(outputs[0], str)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_transcribe_multiple_tensor(self, test_data_dir):
        model = ASRModel.from_pretrained("stt_en_conformer_ctc_small")

        # Load audio file
        import soundfile as sf

        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")
        audio, sr = sf.read(audio_file, dtype='float32')

        audio_file_2 = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an104-mrcb-b.wav")
        audio_2, sr = sf.read(audio_file_2, dtype='float32')
        # Mix second audio to torch.tensor()
        audio_2 = torch.tensor(audio_2)

        # Numpy array test
        outputs = model.transcribe([audio, audio_2], batch_size=2)
        assert len(outputs) == 2
        assert isinstance(outputs[0], str)
        assert isinstance(outputs[1], str)
