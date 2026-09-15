# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import os
import types

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo.collections.speechlm2.parts import augmentation as aug_module
from nemo.collections.speechlm2.parts.augmentation import AudioAugmenter

SR = 16000


class _FakeMeter:
    """Stand-in for ``pyloudnorm.Meter`` with fully deterministic loudness values.

    Any all-(near-)zero signal is reported as ``-inf`` loudness (matching real
    pyloudnorm's behaviour for silent audio); everything else gets a fixed,
    finite value so the test does not depend on the real algorithm.
    """

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def integrated_loudness(self, data):
        if float(np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2))) < 1e-9:
            return float("-inf")
        return -20.0


def _fake_normalize_loudness(data, input_loudness, target_loudness):
    # Distinctive, easy-to-detect transform: flip sign and scale by 3.
    return data * -3.0


@pytest.fixture
def fake_pyloudnorm(monkeypatch):
    """Force the loudness-normalization branch to be live and deterministic,
    without requiring the real (optional) ``pyloudnorm`` package."""
    fake_pyln = types.SimpleNamespace(
        Meter=_FakeMeter,
        normalize=types.SimpleNamespace(loudness=_fake_normalize_loudness),
    )
    monkeypatch.setattr(aug_module, "pyln", fake_pyln)
    monkeypatch.setattr(aug_module, "PYLOUDNORM_AVAILABLE", True)
    return fake_pyln


@pytest.fixture
def roomir_folder(tmp_path):
    # A near-identity impulse response is enough to exercise the convolution path.
    ir = np.zeros(64, dtype=np.float32)
    ir[0] = 1.0
    path = tmp_path / "roomir"
    path.mkdir()
    sf.write(os.path.join(path, "ir.wav"), ir, SR)
    return str(path)


def _make_batch(*, leading_silence: bool) -> torch.Tensor:
    t = torch.arange(SR, dtype=torch.float32) / SR
    row1 = 0.5 * torch.sin(2 * torch.pi * 440 * t)
    row2 = 0.2 * torch.sin(2 * torch.pi * 220 * t)
    if leading_silence:
        row0 = torch.zeros(SR, dtype=torch.float32)
    else:
        row0 = 0.3 * torch.sin(2 * torch.pi * 330 * t)
    return torch.stack([row0, row1, row2], dim=0)


@pytest.mark.unit
def test_room_ir_loudness_norm_does_not_leak_across_batch_rows(fake_pyloudnorm, roomir_folder):
    """A silent row must not disable loudness normalization for the OTHER rows
    in the same batch.

    Regression test for a state leak in ``add_room_ir_to_batch``: the
    ``use_loudness_norm`` function parameter was reassigned to ``False``
    in-place whenever one row's loudness was ``-inf`` (silent audio), and
    that reassignment then applied to every subsequent row in the loop
    because the loop reused the parameter itself instead of a per-row local.
    """
    augmenter = AudioAugmenter(sample_rate=SR)
    audio_lens = torch.tensor([SR, SR, SR])

    calls = {"n": 0}
    real_normalize = aug_module.pyln.normalize.loudness

    def counting_normalize(data, input_loudness, target_loudness):
        calls["n"] += 1
        return real_normalize(data, input_loudness, target_loudness)

    aug_module.pyln.normalize.loudness = counting_normalize

    batch = _make_batch(leading_silence=True)
    augmenter.add_room_ir_to_batch(batch.clone(), audio_lens, roomir_folder, use_loudness_norm=True)

    # Row 0 is silent, so it correctly skips loudness normalization (1 skip).
    # Rows 1 and 2 are normal, non-silent audio and MUST still be
    # loudness-normalized: exactly 2 calls expected.
    assert calls["n"] == 2, (
        f"expected loudness normalization to run for the 2 non-silent rows, got {calls['n']} calls "
        "-- a silent row elsewhere in the batch must not disable it for the others"
    )


@pytest.mark.unit
def test_mic_ir_loudness_norm_does_not_leak_across_batch_rows(fake_pyloudnorm, roomir_folder):
    """Same defect, same fix, in the ``add_mic_ir_to_batch`` sibling."""
    augmenter = AudioAugmenter(sample_rate=SR)
    audio_lens = torch.tensor([SR, SR, SR])

    calls = {"n": 0}
    real_normalize = aug_module.pyln.normalize.loudness

    def counting_normalize(data, input_loudness, target_loudness):
        calls["n"] += 1
        return real_normalize(data, input_loudness, target_loudness)

    aug_module.pyln.normalize.loudness = counting_normalize

    batch = _make_batch(leading_silence=True)
    augmenter.add_mic_ir_to_batch(batch.clone(), audio_lens, roomir_folder, use_loudness_norm=True)

    assert calls["n"] == 2, (
        f"expected loudness normalization to run for the 2 non-silent rows, got {calls['n']} calls "
        "-- a silent row elsewhere in the batch must not disable it for the others"
    )
