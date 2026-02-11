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

from nemo.collections.tts.models import FastPitchModel, HifiGanModel


@pytest.fixture()
def fastpitch_model():
    model = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")
    model.export_config['enable_volume'] = True
    # model.export_config['enable_ragged_batches'] = True
    return model


@pytest.fixture()
def hifigan_model():
    model = HifiGanModel.from_pretrained(model_name="tts_en_hifigan")
    return model


class TestExportable:
    @pytest.mark.pleasefixme
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_FastPitchModel_export_to_onnx(self, fastpitch_model):
        model = fastpitch_model.cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'fp.onnx')
            model.export(output=filename, verbose=True, onnx_opset_version=14, check_trace=True, use_dynamo=True)

    @pytest.mark.pleasefixme
    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_HifiGanModel_export_to_onnx(self, hifigan_model):
        model = hifigan_model.cuda()
        assert hifigan_model.generator is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'hfg.onnx')
            model.export(output=filename, use_dynamo=True, verbose=True, check_trace=True)
