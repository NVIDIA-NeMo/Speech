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


# ---------------------------------------------------------------------------
# validation MTP acceptance metrics
# ---------------------------------------------------------------------------


def test_validation_epoch_end_logs_mtp_acceptance():
    '''on_validation_epoch_end reports per-head acceptance probability and the
    expected acceptance length (always-accepted main token + cumulative product
    of per-head accept probabilities).'''
    from collections import defaultdict

    model = _bare_model()
    model._get_moe_dp_group = lambda: None  # single-rank: _reduce_validation_metric_sums is a no-op
    model.lss_loss = None

    logged = {}
    model.log = lambda name, value, **kwargs: logged.__setitem__(name, float(value))

    # Standard val metrics still need populating — the method aggregates them first.
    model._partial_val_loss_sums = defaultdict(list, {'ds': [torch.tensor(4.0)]})
    model._partial_val_corrects = defaultdict(list, {'ds': [torch.tensor(8.0)]})
    model._partial_val_num_frames = defaultdict(list, {'ds': [torch.tensor(10.0)]})
    model._partial_val_lss = defaultdict(list)

    # Head 1: 8/10 accepted (p1 = 0.8); head 2: 5/10 accepted (p2 = 0.5).
    model._partial_val_mtp_correct = defaultdict(list, {'ds': [torch.tensor([8.0, 5.0])]})
    model._partial_val_mtp_valid = defaultdict(list, {'ds': [torch.tensor([10.0, 10.0])]})

    SALMAutomodel.on_validation_epoch_end(model)

    assert logged['val_mtp_acc_ds/head_1'] == pytest.approx(0.8)
    assert logged['val_mtp_acc_ds/head_2'] == pytest.approx(0.5)
    # 1 (main) + 0.8 + 0.8 * 0.5 = 2.2
    assert logged['val_mtp_accept_length_ds'] == pytest.approx(2.2)
    assert logged['val_mtp_accept_length'] == pytest.approx(2.2)


def test_calculate_mtp_acceptance_with_heads_counts(monkeypatch):
    '''_calculate_mtp_acceptance_with_heads compares each head's argmax against the
    same rolled/masked targets used by the MTP loss and returns per-head
    (correct, valid) counts.'''
    # The helper imports these specific Automodel submodules; importorskip each so the test
    # skips cleanly when the installed Automodel rev predates them (rather than hard-failing).
    pytest.importorskip('nemo_automodel.components.loss.utils', reason='needs Automodel _get_lm_head_module')
    mtp_mod = pytest.importorskip('nemo_automodel.components.models.common.mtp', reason='needs Automodel roll_tensor')
    roll_tensor = mtp_mod.roll_tensor

    from nemo.collections.speechlm2.models.salm_automodel import _calculate_mtp_acceptance_with_heads

    # Identity lm_head over a V==H one-hot space so argmax(hidden) == predicted token id.
    V = 4
    lm_head = torch.nn.Linear(V, V, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(V))
    model = torch.nn.Module()
    model.lm_head = lm_head
    # Avoid depending on Automodel's lm-head discovery internals.
    monkeypatch.setattr('nemo_automodel.components.loss.utils._get_lm_head_module', lambda m: m.lm_head)

    labels = torch.tensor([[0, 1, 2, 3, 0]])  # (1, T)
    T = labels.shape[-1]
    D = 2
    # Every head predicts token 0 everywhere (one-hot index 0).
    one_hot_zero = torch.zeros(1, T, V)
    one_hot_zero[..., 0] = 1.0
    mtp_h = [one_hot_zero.clone() for _ in range(D)]

    correct, valid = _calculate_mtp_acceptance_with_heads(mtp_per_depth_h=mtp_h, labels=labels, model=model)

    # Independently recompute the rolled/masked targets with the real roll_tensor.
    cur = labels
    for k in range(D):
        cur = roll_tensor(cur, shifts=-1, dim=-1)
        masked = cur.clone()
        masked[..., -min(k + 1, T) :] = -100
        valid_mask = masked != -100
        exp_valid = int(valid_mask.sum())
        exp_correct = int(((masked == 0) & valid_mask).sum())  # preds are all 0
        assert int(valid[k]) == exp_valid
        assert int(correct[k]) == exp_correct
