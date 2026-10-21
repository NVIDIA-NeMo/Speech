# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip('utmosv2')
from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

pytestmark = pytest.mark.unit


def test_utmos_resamples_true_lengths_then_batches_features(monkeypatch):
    received = []
    batches = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._cfg = SimpleNamespace(sr=16000, dataset=SimpleNamespace(remove_silent_section=False), phase='test')

        def forward(self, features):
            batches.append(features.shape[0])
            return features[:, :1]

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, cfg, waveforms):
            assert cfg.sr == 16000
            assert not cfg.dataset.remove_silent_section
            assert all(isinstance(wav, np.ndarray) and wav.dtype == np.float32 for wav in waveforms)
            received.extend(waveforms)
            self.waveforms = waveforms

        def __len__(self):
            return len(self.waveforms)

        def __getitem__(self, index):
            return torch.tensor([float(len(self.waveforms[index]))]), torch.tensor(0.0)

    scope = UTMOSv2Calculator.score_waveforms.__wrapped__.__globals__
    monkeypatch.setitem(scope, 'utmosv2', SimpleNamespace(__version__='1.3.0'))
    monkeypatch.setitem(scope, '_UTMOSWaveformDataset', Dataset)
    calculator = UTMOSv2Calculator.__new__(UTMOSv2Calculator)
    calculator.model = Model()
    calculator.device = torch.device('cpu')
    scores = calculator.score_waveforms([torch.ones(n) for n in (2400, 4800, 7200)], 24000, batch_size=2)
    assert [len(wav) for wav in received] == [1600, 3200, 4800]
    assert batches == [2, 1]
    assert scores == [1600, 3200, 4800]
    assert not calculator.model._cfg.dataset.remove_silent_section


def test_utmos_checks_version_and_batch_size(monkeypatch):
    calculator = UTMOSv2Calculator.__new__(UTMOSv2Calculator)
    scope = UTMOSv2Calculator.score_waveforms.__wrapped__.__globals__
    monkeypatch.setitem(scope, 'utmosv2', SimpleNamespace(__version__='1.2.1'))
    with pytest.raises(RuntimeError, match='1.3.0'):
        calculator.score_waveforms([torch.ones(100)], 16000)
    with pytest.raises(ValueError):
        calculator.score_waveforms([torch.ones(100)], 16000, batch_size=0)
    assert calculator.score_waveforms([], 16000) == []
