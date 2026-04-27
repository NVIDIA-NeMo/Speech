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

"""Unit tests for SALM (HF-path) LLM activation checkpointing.

These tests cover:
  * ``set_hf_llm_activation_checkpointing`` — the parallel.py helper that
    wraps each transformer block in ``llm.model.layers`` with
    ``checkpoint_wrapper``.
  * ``SALM.configure_model`` — verifying that the cfg flags
    ``activation_checkpointing_llm`` and ``activation_checkpointing_perception``
    plumb through to the helper and to the perception module respectively.

The tests use lightweight fakes (no HF download / no ASR checkpoint) so they
run on CPU in CI without any heavy dependencies.
"""

from unittest.mock import MagicMock

import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo.collections.speechlm2.models.salm import SALM
from nemo.collections.speechlm2.parts.parallel import set_hf_llm_activation_checkpointing


class _FakeDecoderLayer(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):  # pragma: no cover — forward not exercised
        return self.linear(x)


class _FakeHFLLM(nn.Module):
    """Mimics the structure of a HuggingFace causal-LM that ``SALM`` loads via
    ``transformers.AutoModelForCausalLM``: ``llm.model.layers`` is a
    ``ModuleList`` of decoder blocks. Just enough surface to test the
    wrapping helper without pulling real transformer weights."""

    def __init__(self, num_layers: int = 3):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([_FakeDecoderLayer() for _ in range(num_layers)])
        self.model = inner


class _FakeInnerLLM(nn.Module):
    """Mimics the structure used by the Duplex* models, which extract the
    inner ``model`` from the HF ForCausalLM at construction time, leaving
    ``self.llm.layers`` directly accessible."""

    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer() for _ in range(num_layers)])


# ---------------------------------------------------------------------------
# set_hf_llm_activation_checkpointing
# ---------------------------------------------------------------------------


class TestSetHFLLMActivationCheckpointing:
    def test_noop_when_disabled(self):
        llm = _FakeHFLLM(num_layers=3)
        original_layers = list(llm.model.layers)

        set_hf_llm_activation_checkpointing(llm, enabled=False)

        for i, layer in enumerate(original_layers):
            assert llm.model.layers[i] is layer
            assert not isinstance(llm.model.layers[i], CheckpointWrapper)

    def test_wraps_all_layers_when_enabled(self):
        llm = _FakeHFLLM(num_layers=4)

        set_hf_llm_activation_checkpointing(llm, enabled=True)

        assert len(llm.model.layers) == 4
        for layer in llm.model.layers:
            assert isinstance(layer, CheckpointWrapper)

    def test_wraps_layers_when_inner_llm_layout(self):
        """Duplex* models extract the inner ``model``, so layers live directly
        at ``llm.layers``. The helper must handle this layout too."""
        llm = _FakeInnerLLM(num_layers=3)

        set_hf_llm_activation_checkpointing(llm, enabled=True)

        for layer in llm.layers:
            assert isinstance(layer, CheckpointWrapper)

    def test_inner_llm_layout_noop_when_disabled(self):
        llm = _FakeInnerLLM(num_layers=3)
        original_layers = list(llm.layers)

        set_hf_llm_activation_checkpointing(llm, enabled=False)

        for i, layer in enumerate(original_layers):
            assert llm.layers[i] is layer
            assert not isinstance(llm.layers[i], CheckpointWrapper)

    def test_noop_when_llm_has_no_model_attr(self):
        """An architecture without ``.model.layers`` (e.g. an unrecognised
        backbone) should not raise — the helper degrades gracefully."""

        class ForeignLLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.something = nn.Linear(4, 4)

        llm = ForeignLLM()
        set_hf_llm_activation_checkpointing(llm, enabled=True)  # must not raise
        assert not isinstance(llm.something, CheckpointWrapper)

    def test_noop_when_model_has_no_layers(self):
        class LLMWithEmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()

        llm = LLMWithEmptyModel()
        set_hf_llm_activation_checkpointing(llm, enabled=True)  # must not raise

    def test_wrapped_layer_forward_runs(self):
        """Sanity: a wrapped layer should still be callable. We don't validate
        gradient-checkpoint behavior (that's torch's responsibility), only that
        wrapping doesn't break the forward path."""
        llm = _FakeHFLLM(num_layers=1)
        set_hf_llm_activation_checkpointing(llm, enabled=True)

        x = torch.randn(1, 4, requires_grad=True)
        y = llm.model.layers[0](x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# SALM.configure_model — cfg-driven AC plumbing
# ---------------------------------------------------------------------------


def _make_fake_salm(cfg: dict) -> SALM:
    """Build a bare-bones SALM instance that bypasses the real ``__init__``.

    We only set the attributes that ``configure_model`` touches so we can
    exercise the AC plumbing without loading TinyLlama / Canary checkpoints.
    """
    model = SALM.__new__(SALM)
    nn.Module.__init__(model)
    model.cfg = DictConfig(cfg)
    model.llm = _FakeHFLLM(num_layers=3)
    model.perception = MagicMock()
    model._device_mesh = None  # short-circuits TP/FSDP code paths
    model._trainer = None
    model._use_fsdp = False
    model._use_tp = False
    return model


class TestSALMConfigureModelActivationCheckpointing:
    def test_defaults_are_disabled(self):
        """Without any AC keys in cfg, both perception and LLM must remain
        unwrapped — the change should be backwards-compatible."""
        model = _make_fake_salm({})
        original_layers = list(model.llm.model.layers)

        model.configure_model()

        model.perception.set_activation_checkpointing.assert_called_once_with(False)
        for i, layer in enumerate(original_layers):
            assert model.llm.model.layers[i] is layer
            assert not isinstance(model.llm.model.layers[i], CheckpointWrapper)

    def test_explicit_false_keeps_layers_unwrapped(self):
        model = _make_fake_salm(
            {
                "activation_checkpointing_llm": False,
                "activation_checkpointing_perception": False,
            }
        )

        model.configure_model()

        model.perception.set_activation_checkpointing.assert_called_once_with(False)
        for layer in model.llm.model.layers:
            assert not isinstance(layer, CheckpointWrapper)

    def test_llm_flag_wraps_transformer_layers(self):
        model = _make_fake_salm({"activation_checkpointing_llm": True})

        model.configure_model()

        for layer in model.llm.model.layers:
            assert isinstance(layer, CheckpointWrapper)
        # Perception flag defaulted to False.
        model.perception.set_activation_checkpointing.assert_called_once_with(False)

    def test_perception_flag_routes_to_set_activation_checkpointing(self):
        model = _make_fake_salm({"activation_checkpointing_perception": True})

        model.configure_model()

        model.perception.set_activation_checkpointing.assert_called_once_with(True)
        # LLM flag defaulted to False — layers must remain unwrapped.
        for layer in model.llm.model.layers:
            assert not isinstance(layer, CheckpointWrapper)

    def test_both_flags_independently(self):
        model = _make_fake_salm(
            {
                "activation_checkpointing_llm": True,
                "activation_checkpointing_perception": True,
            }
        )

        model.configure_model()

        model.perception.set_activation_checkpointing.assert_called_once_with(True)
        for layer in model.llm.model.layers:
            assert isinstance(layer, CheckpointWrapper)

    def test_ac_runs_before_fsdp_short_circuit(self):
        """The AC application must happen even when ``device_mesh is None``
        (i.e. before the early return that skips TP/FSDP). This guarantees
        that AC is applied during single-GPU/CPU runs too, and — critically
        for distributed runs — that wrapping happens before fully_shard."""
        model = _make_fake_salm({"activation_checkpointing_llm": True})
        assert model._device_mesh is None  # precondition

        model.configure_model()

        for layer in model.llm.model.layers:
            assert isinstance(layer, CheckpointWrapper)
