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

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader, Dataset

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
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


class DummyDatasetAudioOnly(Dataset):
    def __init__(self, audio_files: List[str], config: Dict):
        self.audio_files = audio_files
        self.config = config

    def __getitem__(self, index):
        data = self.audio_files[index]
        data = torch.tensor([float(data)]).view(1)
        return data

    def __len__(self):
        return len(self.audio_files)


class DummyDataset(Dataset):
    def __init__(self, audio_tensors: List[str], config: Dict = None):
        self.audio_tensors = audio_tensors
        self.config = config

    def __getitem__(self, index):
        data = self.audio_tensors[index]
        samples = torch.tensor(data)
        # Calculate seq length
        seq_len = torch.tensor(samples.shape[0], dtype=torch.long)

        # Dummy text tokens
        text_tokens = torch.tensor([0], dtype=torch.long)
        text_tokens_len = torch.tensor(1, dtype=torch.long)

        return (samples, seq_len, text_tokens, text_tokens_len)

    def __len__(self):
        return len(self.audio_tensors)


@pytest.mark.with_downloads()
@pytest.fixture()
def audio_files(test_data_dir):
    """
    Returns a list of audio files for testing.
    """
    import soundfile as sf

    audio_file1 = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")
    audio_file2 = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an104-mrcb-b.wav")

    audio1, _ = sf.read(audio_file1, dtype='float32')
    audio2, _ = sf.read(audio_file2, dtype='float32')

    return audio1, audio2


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
        dataset = DummyDatasetAudioOnly(config['paths2audio_files'], config)

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
    def test_transcribe_return_hypothesis(self, test_data_dir, fast_conformer_ctc_model):
        audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")

        # Audio file test
        outputs = fast_conformer_ctc_model.transcribe(audio_file, batch_size=1, return_hypotheses=True)
        assert len(outputs) == 1
        assert isinstance(outputs[0], Hypothesis)

        hyp = outputs[0]
        assert isinstance(hyp.text, str)
        assert isinstance(hyp.y_sequence, torch.Tensor)
        assert isinstance(hyp.alignments, torch.Tensor)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_transcribe_tensor(self, audio_files, fast_conformer_ctc_model):

        audio, _ = audio_files
        # Numpy array test
        outputs = fast_conformer_ctc_model.transcribe(audio, batch_size=1)
        assert len(outputs) == 1
        assert isinstance(outputs[0], Hypothesis)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_transcribe_multiple_tensor(self, audio_files, fast_conformer_ctc_model):

        audio, audio_2 = audio_files
        # Mix second audio to torch.tensor()
        audio_2 = torch.tensor(audio_2)

        # Numpy array test
        outputs = fast_conformer_ctc_model.transcribe([audio, audio_2], batch_size=2)
        assert len(outputs) == 2
        assert isinstance(outputs[0], Hypothesis)
        assert isinstance(outputs[1], Hypothesis)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_transcribe_dataloader(self, audio_files, fast_conformer_ctc_model):

        audio, audio2 = audio_files

        dataset = DummyDataset([audio, audio2])
        collate_fn = lambda x: _speech_collate_fn(x, pad_id=0)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

        # DataLoader test
        outputs = fast_conformer_ctc_model.transcribe(dataloader, batch_size=1)
        assert len(outputs) == 2
        assert isinstance(outputs[0], Hypothesis)
        assert isinstance(outputs[1], Hypothesis)

    @pytest.mark.unit
    def test_transcribe_return_nbest_rnnt(self, audio_files, fast_conformer_transducer_model):
        fast_conformer_transducer_model.eval()
        audio1, audio2 = audio_files

        orig_decoding_config = copy.deepcopy(fast_conformer_transducer_model.cfg.decoding)

        decoding_config = copy.deepcopy(fast_conformer_transducer_model.cfg.decoding)
        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypothesis"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False
        fast_conformer_transducer_model.change_decoding_strategy(decoding_config)

        outputs = fast_conformer_transducer_model.transcribe([audio1, audio2], batch_size=1, timestamps=False)

        assert len(outputs) == 2
        assert all(len(output) >= 1 for output in outputs)
        assert all(isinstance(output, list) for output in outputs)
        assert all(isinstance(hyp, Hypothesis) for output in outputs for hyp in output)

        # Reset the decoding strategy to original
        fast_conformer_transducer_model.change_decoding_strategy(orig_decoding_config)

    @pytest.mark.unit
    def test_transcribe_return_nbest_canary(self, audio_files, canary_1b_flash):
        canary_1b_flash.eval()
        audio1, audio2 = audio_files

        orig_decoding_config = copy.deepcopy(canary_1b_flash.cfg.decoding)

        decoding_config = copy.deepcopy(canary_1b_flash.cfg.decoding)
        with open_dict(decoding_config):
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypothesis"] = False
        canary_1b_flash.change_decoding_strategy(decoding_config)

        outputs = canary_1b_flash.transcribe([audio1, audio2], batch_size=1, timestamps=False)

        assert len(outputs) == 2
        assert all(len(output) >= 1 for output in outputs)
        assert all(isinstance(output, list) for output in outputs)
        assert all(isinstance(hyp, Hypothesis) for output in outputs for hyp in output)

        # Reset the decoding strategy to original
        canary_1b_flash.change_decoding_strategy(orig_decoding_config)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_timestamps_with_transcribe(self, audio_files, fast_conformer_ctc_model):
        audio1, audio2 = audio_files

        output = fast_conformer_ctc_model.transcribe([audio1, audio2], timestamps=True)

        # check len of output
        assert len(output) == 2

        # check hypothesis object
        assert isinstance(output[0], Hypothesis)
        # check transcript
        assert output[0].text == 'stop'
        assert output[1].text == 'start'

        # check timestamp
        assert output[0].timestamp['segment'][0]['start'] == pytest.approx(0.4)
        assert output[0].timestamp['segment'][0]['end'] == pytest.approx(0.48)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_timestamps_with_transcribe_hybrid(self, audio_files, fast_conformer_hybrid_model):
        audio1, audio2 = audio_files

        output = fast_conformer_hybrid_model.transcribe([audio1, audio2], timestamps=True)

        # check len of output
        assert len(output) == 2

        # check hypothesis object
        assert isinstance(output[0], Hypothesis)
        # check transcript
        assert output[0].text == 'Stop?'
        assert output[1].text == 'Start.'

        # check timestamp
        assert output[0].timestamp['segment'][0]['start'] == pytest.approx(0.48)
        assert output[0].timestamp['segment'][0]['end'] == pytest.approx(0.72)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_timestamps_with_transcribe_hybrid_ctc_head(self, audio_files, fast_conformer_hybrid_model):
        audio1, audio2 = audio_files

        fast_conformer_hybrid_model.change_decoding_strategy(decoder_type='ctc')

        output = fast_conformer_hybrid_model.transcribe([audio1, audio2], timestamps=True)

        # check len of output
        assert len(output) == 2

        # check hypothesis object
        assert isinstance(output[0], Hypothesis)
        # check transcript
        assert output[0].text in ['Stop', 'Stop?']
        assert output[1].text in ['Start', 'Start.']

        # check timestamp
        assert output[0].timestamp['segment'][0]['start'] == pytest.approx(0.4)
        assert output[0].timestamp['segment'][0]['end'] == pytest.approx(0.72)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_timestamps_with_transcribe_canary_flash(self, audio_files, canary_1b_flash):
        audio1, audio2 = audio_files

        output = canary_1b_flash.transcribe([audio1, audio2], timestamps=True)

        # check len of output
        assert len(output) == 2

        # check hypothesis object
        assert isinstance(output[0], Hypothesis)
        # check transcript
        assert output[0].text == 'Stop'
        assert output[1].text == 'start'

        # check timestamp
        assert output[0].timestamp['segment'][0]['start'] == pytest.approx(0.32)
        assert output[0].timestamp['segment'][0]['end'] == pytest.approx(0.72)
