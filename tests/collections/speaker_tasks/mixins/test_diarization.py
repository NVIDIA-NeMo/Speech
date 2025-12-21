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

# pyright: reportMissingImports=false
# pylint: disable=import-error,redefined-outer-name


import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig, SpkDiarizationMixin


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        # Make the test deterministic and ensure positive outputs.
        with torch.no_grad():
            self.encoder.weight.fill_(1.0)
            self.encoder.bias.fill_(1.0)

        self.execution_count = 0
        self.flag_begin = False

    def forward(self, x):
        # Input: [1, 1] Output = [1, 1
        out = self.encoder(x)
        return out


class AudioPathDataset(Dataset):
    def __init__(self, audio_filepaths: List[str]):
        self._audio_filepaths = audio_filepaths

    def __len__(self) -> int:
        return len(self._audio_filepaths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        samples, _ = sf.read(self._audio_filepaths[index], dtype="float32", always_2d=False)
        if hasattr(samples, "ndim") and samples.ndim == 2:
            samples = samples.mean(axis=1)
        waveform = torch.as_tensor(samples, dtype=torch.float32)
        length = torch.tensor(waveform.shape[0], dtype=torch.long)
        return waveform, length


def collate(batch):
    waveforms, lengths = zip(*batch)
    padded = pad_sequence(waveforms, batch_first=True)  # (B, T)
    lengths_tensor = torch.stack(lengths, dim=0)
    return padded, lengths_tensor


@pytest.fixture()
def audio_files(test_data_dir):
    """
    Returns audio arrays + sample rate + filepaths for testing.
    """

    audio_file1 = os.path.join(test_data_dir, "an4_speaker", "an4", "wav", "an4_clstk", "fash", "an251-fash-b.wav")
    audio_file2 = os.path.join(test_data_dir, "an4_speaker", "an4", "wav", "an4_clstk", "ffmm", "cen1-ffmm-b.wav")

    audio1, sample_rate1 = sf.read(audio_file1, dtype='float32', always_2d=False)
    audio2, sample_rate2 = sf.read(audio_file2, dtype='float32', always_2d=False)
    assert int(sample_rate1) == int(sample_rate2)

    return audio1, audio2, int(sample_rate1), audio_file1, audio_file2


class DiarizableDummy(DummyModel, SpkDiarizationMixin):
    def _diarize_on_begin(self, audio, diarcfg: DiarizeConfig):
        super()._diarize_on_begin(audio, diarcfg)
        self.flag_begin = True

    def _setup_diarize_dataloader(self, config: Dict) -> DataLoader:
        if "manifest_filepath" in config:
            filepaths: List[str] = []
            with open(config["manifest_filepath"], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    filepaths.append(json.loads(line)["audio_filepath"])
        else:
            filepaths = list(config["paths2audio_files"])

        return DataLoader(
            dataset=AudioPathDataset(filepaths),
            batch_size=int(config.get("batch_size", 1)),
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=False,
            drop_last=False,
            collate_fn=collate,
        )

    def _diarize_forward(self, batch: Any):
        """
        Real inference step for diarization tests.

        The dataloader yields `(padded_waveforms, lengths)` where:
        - padded_waveforms: (B, T)
        - lengths: (B,)

        We compute a masked mean per sample -> (B, 1) and run it through the model.
        """
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise TypeError(f"Expected batch=(waveforms, lengths), got: {type(batch)}")

        waveforms, lengths = batch
        if waveforms.dim() != 2:
            raise ValueError(f"Expected waveforms of shape (B, T), got {tuple(waveforms.shape)}")
        if lengths.dim() != 1:
            raise ValueError(f"Expected lengths of shape (B,), got {tuple(lengths.shape)}")

        # Masked mean pooling over time dimension.
        _, T = waveforms.shape
        device = waveforms.device
        t = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        mask = (t < lengths.unsqueeze(1)).to(waveforms.dtype)  # (B, T)
        denom = lengths.to(waveforms.dtype).clamp_min(1.0).unsqueeze(1)  # (B, 1)
        pooled = (waveforms * mask).sum(dim=1, keepdim=True) / denom  # (B, 1)

        preds = self(pooled)  # (B, 1)
        return preds

    def _diarize_output_processing(self, outputs, uniq_ids, diarcfg: DiarizeConfig):
        self.execution_count += 1
        # Ensure "one scalar per input sample".
        outputs = outputs.detach().cpu().view(outputs.shape[0], -1).mean(dim=1)
        return [float(x) for x in outputs]


@pytest.fixture()
def dummy_model():
    return DiarizableDummy()


class TestSpkDiarizationMixin:
    pytestmark = pytest.mark.with_downloads()

    @pytest.mark.unit
    def test_constructor_non_instance(self):
        model = DummyModel()
        assert not isinstance(model, SpkDiarizationMixin)
        assert not hasattr(model, 'diarize')

    @pytest.mark.unit
    def test_diarize_wav_path_single(self, dummy_model, audio_files):
        dummy_model = dummy_model.eval()
        _, _, _, audio_file1, _ = audio_files
        outputs = dummy_model.diarize(audio_file1, batch_size=1)
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert outputs[0] > 0

    @pytest.mark.unit
    def test_diarize_wav_path_list(self, dummy_model, audio_files):
        dummy_model = dummy_model.eval()
        _, _, _, audio_file1, audio_file2 = audio_files
        outputs = dummy_model.diarize([audio_file1, audio_file2], batch_size=1)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
        assert outputs[0] > 0
        assert outputs[1] > 0

    @pytest.mark.unit
    def test_diarize_manifest_jsonl_path(self, dummy_model, audio_files, tmp_path: Path):
        dummy_model = dummy_model.eval()
        _, _, _, audio_file1, audio_file2 = audio_files
        manifest_path = tmp_path / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for audio_file in (audio_file1, audio_file2):
                f.write(json.dumps({"audio_filepath": audio_file, "offset": 0, "duration": None, "text": "-"}) + "\n")
        outputs = dummy_model.diarize(str(manifest_path), batch_size=1)
        assert isinstance(outputs, list)
        assert len(outputs) == 2

    @pytest.mark.unit
    def test_diarize_numpy_single_requires_sample_rate(self, dummy_model, audio_files):
        dummy_model = dummy_model.eval()
        audio1, _, _, _, _ = audio_files
        with pytest.raises(ValueError):
            _ = dummy_model.diarize(audio1, batch_size=1)

    @pytest.mark.unit
    def test_diarize_numpy_single(self, dummy_model, audio_files):
        dummy_model = dummy_model.eval()
        audio1, _, sample_rate, _, _ = audio_files
        outputs = dummy_model.diarize(audio1, batch_size=1, sample_rate=sample_rate)
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert outputs[0] > 0

    @pytest.mark.unit
    def test_diarize_numpy_list(self, dummy_model, audio_files):
        dummy_model = dummy_model.eval()
        audio1, audio2, sample_rate, _, _ = audio_files
        outputs = dummy_model.diarize([audio1, audio2], batch_size=1, sample_rate=sample_rate)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
        assert outputs[0] > 0
        assert outputs[1] > 0

    @pytest.mark.unit
    def test_diarize_numpy_list_but_no_sample_rate(self, dummy_model, audio_files):
        dummy_model = dummy_model.eval()
        # Numpy audio inputs require sample_rate; the mixin should raise with a clear message.
        with pytest.raises(
            ValueError, match=r"Sample rate is not set\. Numpy audio inputs require sample_rate to be set\."
        ):
            _ = dummy_model.diarize(audio_files, batch_size=1)

    @pytest.mark.unit
    def test_transribe_override_config_incorrect(self, dummy_model, audio_files):
        # Not subclassing DiarizeConfig
        @dataclass
        class OverrideConfig:
            batch_size: int = 1
            output_type: str = 'dict'

        dummy_model = dummy_model.eval()

        audio1, _, _, _, _ = audio_files
        override_cfg = OverrideConfig(batch_size=1, output_type='dict')
        with pytest.raises(ValueError):
            _ = dummy_model.diarize(audio1, override_config=override_cfg)

    @pytest.mark.unit
    def test_transribe_override_config_correct(self, dummy_model, audio_files):
        @dataclass
        class OverrideConfig(DiarizeConfig):
            output_type: str = 'dict'
            verbose: bool = False

        dummy_model = dummy_model.eval()
        audio1, _, sample_rate, _, _ = audio_files
        override_cfg = OverrideConfig(batch_size=1, output_type='list', sample_rate=sample_rate)
        outputs = dummy_model.diarize(audio1, override_config=override_cfg)

        assert isinstance(outputs, list)
        assert len(outputs) == 1
