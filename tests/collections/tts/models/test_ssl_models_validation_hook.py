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

"""
Regression test for FastPitchModel_SSL / SSLDisentangler's `on_validation_epoch_end` hook.

Lightning's evaluation loop calls `on_validation_epoch_end` with no positional arguments
(`lightning.pytorch.loops.evaluation_loop._EvaluationLoop._on_evaluation_epoch_end` does
`call._call_lightning_module_hook(trainer, "on_validation_epoch_end")`), and has done so since the
PL 2.0 migration (#6433), which removed the `outputs` parameter from every other model in this repo
(e.g. `FastPitchModel.on_validation_epoch_end`, which reads from `self.validation_step_outputs`
instead). These two SSL models kept the pre-2.0 `(self, outputs)` signature, so any real validation
epoch on either model raises a `TypeError` before user code ever runs, aborting training at the very
first sanity-check validation pass.
"""

import pytest
import torch

from nemo.collections.tts.models.fastpitch_ssl import FastPitchModel_SSL
from nemo.collections.tts.models.ssl_tts import SSLDisentangler


def _call_on_validation_epoch_end_like_lightning(model):
    """Invoke the hook exactly the way `_call_lightning_module_hook` does: `fn()`, no arguments."""
    return model.on_validation_epoch_end()


@pytest.mark.unit
def test_fastpitch_model_ssl_on_validation_epoch_end_accepts_no_arguments():
    # `@experimental` wraps the class in a wrapt proxy; unwrap it and bypass __init__ (which needs a
    # full model config) since the hook's argument-binding bug doesn't depend on any of that.
    model = object.__new__(FastPitchModel_SSL.__wrapped__)
    model._trainer = None
    model._fabric = None
    model.pitch_conditioning = False
    model._validation_step_outputs = [
        {
            "val_loss": torch.tensor(1.0),
            "mel_loss": torch.tensor(0.5),
            "mel_target": None,
            "mel_pred": None,
            "spec_len": None,
            "pitch_target": None,
            "pitch_pred": None,
        }
    ]
    logged = {}
    model.log = lambda key, value, *args, **kwargs: logged.__setitem__(key, value)

    _call_on_validation_epoch_end_like_lightning(model)

    assert logged["v_loss"] == torch.tensor(1.0)
    assert logged["v_mel_loss"] == torch.tensor(0.5)
    assert model.validation_step_outputs == []


@pytest.mark.unit
def test_ssl_disentangler_on_validation_epoch_end_accepts_no_arguments():
    model = object.__new__(SSLDisentangler.__wrapped__)
    model._trainer = None
    model._fabric = None
    model._validation_step_outputs = [
        {
            "val_loss": torch.tensor(1.0),
            "sv_loss": torch.tensor(0.2),
            "ctc_loss": torch.tensor(0.3),
            "content_loss": torch.tensor(0.1),
            "accuracy_sv": torch.tensor(80.0),
            "cer": torch.tensor(0.05),
        }
    ]
    logged = {}
    model.log = lambda key, value, *args, **kwargs: logged.__setitem__(key, value)

    _call_on_validation_epoch_end_like_lightning(model)

    assert logged["val_loss"] == torch.tensor(1.0)
    assert logged["cer"] == torch.tensor(0.05)
    assert model.validation_step_outputs == []
