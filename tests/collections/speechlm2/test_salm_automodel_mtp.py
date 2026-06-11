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
import pytest
import torch

from nemo.collections.speechlm2.models import SALMAutomodel


def _bare_model():
    '''Create a SALMAutomodel instance without loading any weights.'''
    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    return model


# ---------------------------------------------------------------------------
# _mtp_enabled property
# ---------------------------------------------------------------------------


def test_mtp_enabled_false_when_llm_missing():
    model = _bare_model()
    assert not model._mtp_enabled


def test_mtp_enabled_false_when_mtp_attr_missing():
    model = _bare_model()
    model.llm = torch.nn.Module()
    assert not model._mtp_enabled


def test_mtp_enabled_false_when_mtp_is_none():
    model = _bare_model()
    model.llm = torch.nn.Module()
    model.llm.mtp = None
    assert not model._mtp_enabled


def test_mtp_enabled_true_when_mtp_attached():
    model = _bare_model()
    model.llm = torch.nn.Module()
    model.llm.mtp = torch.nn.Linear(4, 4)
    assert model._mtp_enabled


# ---------------------------------------------------------------------------
# forward: mtp_per_depth_h extraction
# ---------------------------------------------------------------------------


class _DictLikeOutput(dict):
    '''Mimics a HuggingFace ModelOutput — dict keys are also accessible as attributes.'''

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class _FakeLLM(torch.nn.Module):
    def __init__(self, out):
        super().__init__()
        self._out = out
        self.mtp = None

    def forward(self, **_kwargs):
        return self._out


def _make_forward_model(llm_out):
    '''Minimal model that can run the forward() method with a mocked LLM.'''
    model = _bare_model()
    model.llm = _FakeLLM(llm_out)
    model._use_tp = False
    return model


def test_forward_extracts_mtp_per_depth_h_when_present():
    mtp_h = torch.randn(1, 4, 8)
    fake_out = _DictLikeOutput(logits=torch.randn(1, 4, 32), mtp_per_depth_h=mtp_h)
    model = _make_forward_model(fake_out)

    result = model.forward(
        input_embeds=torch.randn(1, 4, 32),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    assert 'mtp_per_depth_h' in result
    assert result['mtp_per_depth_h'] is mtp_h


def test_forward_omits_mtp_per_depth_h_when_absent():
    fake_out = _DictLikeOutput(logits=torch.randn(1, 4, 32))
    model = _make_forward_model(fake_out)

    result = model.forward(
        input_embeds=torch.randn(1, 4, 32),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    assert 'mtp_per_depth_h' not in result
