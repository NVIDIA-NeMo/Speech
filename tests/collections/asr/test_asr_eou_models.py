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

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.models.asr_eou_models import EncDecHybridRNNTCTCBPEEOUModel, EncDecRNNTBPEEOUModel


class _Config(dict):
    """Minimal stand-in for the model cfg, which is only read through ``.get``."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


class _Joint:
    """``fuse_loss_wer`` defaults to False in RNNTJoint, so that is the branch under test."""

    fuse_loss_wer = False


class _WER:
    def update(self, **kwargs):
        pass

    def get_hypotheses(self):
        return [SimpleNamespace(y_sequence=torch.tensor([1, 2, 3]))]

    def compute(self):
        return torch.tensor(0.5), torch.tensor(1.0), torch.tensor(2.0)

    def reset(self):
        pass


class _Batch:
    audio_signal = torch.zeros(1, 16000)
    audio_lengths = torch.tensor([16000])
    text_tokens = torch.tensor([[1, 2, 3]])
    text_token_lengths = torch.tensor([3])
    sample_ids = [0]
    audio_filepaths = ["audio.wav"]


class _EOUModelStub:
    """Supplies only the attributes ``validation_pass`` touches, so the test exercises the
    real control flow of the model without building encoder/decoder/tokenizer weights."""

    def __init__(self, save_pred_to_file=None):
        self.cfg = _Config(save_pred_to_file=save_pred_to_file)
        self.joint = _Joint()
        self.compute_eval_loss = False
        self.wer = _WER()
        self.ctc_wer = _WER()
        self.ctc_loss_weight = 0.0
        self.model_guid = "test-guid"
        self.trainer = SimpleNamespace(global_step=0)

    def forward(self, input_signal, input_signal_length):
        return torch.zeros(1, 4, 8), torch.tensor([4])

    def ctc_decoder(self, encoder_output):
        return torch.zeros(1, 4, 8)

    def _get_text_from_tokens(self, *args, **kwargs):
        return ["hello"]

    def _get_eou_predictions_from_hypotheses(self, hypotheses, batch):
        return []

    def _calculate_eou_metrics(self, eou_predictions, batch):
        return [], []

    def add_interctc_losses(self, loss_value, *args, **kwargs):
        return loss_value, {}

    def log(self, *args, **kwargs):
        pass


class TestEOUValidationPass:
    @pytest.mark.unit
    @pytest.mark.parametrize("model_cls", [EncDecHybridRNNTCTCBPEEOUModel, EncDecRNNTBPEEOUModel])
    def test_validation_pass_without_save_pred_to_file(self, model_cls):
        """``save_pred_to_file`` is unset by default, so no prediction text is computed and
        none must be logged."""
        model = _EOUModelStub()
        logs = model_cls.validation_pass(model, _Batch(), 0)
        assert 'val_text_pred' not in logs
        assert 'val_wer' in logs

    @pytest.mark.unit
    @pytest.mark.parametrize("model_cls", [EncDecHybridRNNTCTCBPEEOUModel, EncDecRNNTBPEEOUModel])
    def test_validation_pass_with_save_pred_to_file(self, model_cls, tmp_path):
        """With ``save_pred_to_file`` set, the prediction text is still logged."""
        model = _EOUModelStub(save_pred_to_file=str(tmp_path / "preds.json"))
        logs = model_cls.validation_pass(model, _Batch(), 0)
        assert logs['val_text_pred'] == ["hello"]
