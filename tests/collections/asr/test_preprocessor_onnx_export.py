# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Check waveform export across runtime batch sizes and STFT/padding boundaries."""

import numpy as np
import pytest
import torch

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor


@pytest.mark.unit
@pytest.mark.parametrize("pad_to", [0, 16])
def test_dynamic_preprocessor_onnx(tmp_path, pad_to):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(17)
    model = AudioToMelSpectrogramPreprocessor(features=128, n_fft=512, pad_to=pad_to).eval()
    path = tmp_path / "preprocessor.onnx"
    model.export(
        str(path),
        input_example=(torch.randn(2, 16000), torch.tensor([16000, 12000])),
        onnx_opset_version=17,
        dynamic_axes={
            "input_signal": {0: "batch", 1: "samples"},
            "length": {0: "batch"},
            "processed_signal": {0: "batch", 2: "frames"},
            "processed_length": {0: "batch"},
        },
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch, samples in [(1, 2400), (3, 2560), (2, 2720), (2, 16000), (1, 32000), (2, 800)]:
        signal = torch.randn(batch, samples)
        lengths = torch.linspace(0, samples, batch).long() if batch > 1 else torch.tensor([samples])
        signal[torch.arange(samples)[None, :] >= lengths[:, None]] = 0
        expected, expected_lengths = model(input_signal=signal, length=lengths)
        actual, actual_lengths = session.run(None, {"input_signal": signal.numpy(), "length": lengths.numpy()})
        np.testing.assert_array_equal(actual_lengths, expected_lengths.numpy())
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)
        assert np.isfinite(actual).all()
