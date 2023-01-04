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

import os
import tempfile

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.tts.models import WaveGlowModel
from nemo.core.classes import typecheck

mcfg = DictConfig(
    {
        "_target_": "nemo.collections.tts.modules.waveglow.WaveGlowModule",
        "n_flows": 12,
        "n_group": 8,
        "n_mel_channels": 80,
        "n_early_every": 4,
        "n_early_size": 2,
        "n_wn_channels": 512,
        "n_wn_layers": 8,
        "wn_kernel_size": 3,
    }
)

pcfg = DictConfig(
    {
        "_target_": "nemo.collections.asr.parts.preprocessing.features.FilterbankFeatures",
        "dither": 0.0,
        "nfilt": 80,
        "stft_conv": False,
    }
)

wcfg = DictConfig({"waveglow": mcfg, "sigma": 1.0, "preprocessor": pcfg,})


def input_example(sz):
    mel = torch.randn(1, 1, 80, sz).cuda().half()
    z = torch.randn(1, 8, sz * 256 // 8, 1).cuda().half()
    return (
        mel,
        z,
    )


def taco2wg(spec, z):
    spec = spec.permute(0, 3, 2, 1).contiguous()
    return spec.view(spec.size(0), spec.size(1), -1), z.view(z.size(0), z.size(1), -1)


# Wrapper method to convert Jasper's Taco2 output to WG input and call inference
def forward_wrapper(self, spec, z=None):
    spec, z = taco2wg(spec, z)
    audio = self.waveglow.norm_dist_to_audio(spec=spec, sigma=1.0, z=z)
    return audio


class TestWaveGlow:
    @pytest.mark.pleasefixme
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_export_to_onnx(self):
        model = WaveGlowModel(wcfg)
        model = model.cuda().half()
        typecheck.set_typecheck_enabled(enabled=False)
        with tempfile.TemporaryDirectory() as tmpdir, model.nemo_infer():
            tmp_file_name = os.path.join(tmpdir, "waveglow.onnx")

            n_mels = 80
            # Test export.
            inp = input_example(n_mels)
            inp1 = taco2wg(*inp)
            inp2 = inp1
            res1 = model.waveglow(*inp1)
            res2 = model.waveglow(*inp2)
            assert torch.allclose(res1, res2, rtol=0.01, atol=0.1)
            WaveGlowModel.forward_for_export = forward_wrapper
            model.export(
                tmp_file_name, input_example=inp, verbose=False, check_trace=False, do_constant_folding=True,
            )


if __name__ == "__main__":
    t = TestWaveGlow()
    t.test_export_to_onnx()
