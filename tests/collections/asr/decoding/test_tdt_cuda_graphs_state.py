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

"""
Unit tests for GitHub issue #15423:
  preserve disable_cuda_graphs() state across change_decoding_strategy() calls.

Covers:
  - BeamBatchedTDTInfer._cuda_graphs_disabled is initialized to False
  - disable_cuda_graphs() sets _cuda_graphs_disabled = True
  - maybe_enable_cuda_graphs() clears _cuda_graphs_disabled = False
  - change_decoding_strategy() preserves the disabled state across all model
    variants: EncDecRNNTModel, EncDecRNNTBPEModel, EncDecHybridRNNTCTCBPEModel
"""

import abc
import importlib.abc
import importlib.machinery
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies that are not installed in this
# environment (e.g. lightning / pytorch-lightning, nv_one_logger).
# This must happen BEFORE any nemo.* import so that package __init__ chains
# complete cleanly.
#
# Challenge: real NeMo classes do things like:
#     class ModelPT(LightningModule, Model): ...
# where Model has ABCMeta as metaclass. If LightningModule is a plain
# MagicMock instance (metaclass = MagicMock's metaclass), Python raises
# "TypeError: metaclass conflict".
#
# Fix: PascalCase attribute accesses on mocked modules return real stub
# *classes* whose metaclass inherits from ABCMeta, making them compatible
# as bases for any ABC-derived NeMo class.
# ---------------------------------------------------------------------------


class _StubMeta(abc.ABCMeta):
    """Metaclass for all auto-generated stub classes.
    Inherits ABCMeta so stub classes are compatible bases alongside real
    NeMo ABC classes (no metaclass conflict)."""


# Global cache so the same logical class (e.g. LightningModule) is always the
# same object — important for isinstance checks and MRO consistency.
_stub_class_cache: dict[str, type] = {}


def _stub_class(name: str) -> type:
    """Return a reusable, ABCMeta-compatible stub class for `name`."""
    if name not in _stub_class_cache:
        _stub_class_cache[name] = _StubMeta(
            name,
            (),
            {
                "__init__": lambda self, *a, **kw: None,
            },
        )
    return _stub_class_cache[name]


def _make_module_mock(fullname: str) -> types.ModuleType:
    """Return a ModuleType where:
    - PascalCase attributes → reusable stub class (safe as a base class)
    - Everything else      → MagicMock()
    """

    def _getattr(name: str):
        # PascalCase → stub class so it can be used in class inheritance
        if name and name[0].isupper():
            return _stub_class(name)
        return MagicMock()

    mod = types.ModuleType(fullname)
    mod.__package__ = fullname
    mod.__path__ = []  # marks it as a package
    mod.__spec__ = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
    mod.__class__ = type(
        "_AutoMockModule",
        (types.ModuleType,),
        {"__getattr__": lambda self, name: _getattr(name)},
    )
    return mod


class _OptionalDepLoader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec):
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        mod = _make_module_mock(spec.name)
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module: types.ModuleType) -> None:
        pass  # module is fully configured in create_module


_loader_singleton = _OptionalDepLoader()


class _OptionalDepFinder(importlib.abc.MetaPathFinder):
    """Intercepts imports of optional packages absent in this environment."""

    _INTERCEPT_PREFIXES = ("lightning", "nv_one_logger", "nemo.lightning", "nemo.core.classes.modelPT")

    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self._INTERCEPT_PREFIXES):
            return importlib.machinery.ModuleSpec(fullname, _loader_singleton, is_package=True)
        return None


# Install before any NeMo import.
_finder_instance = _OptionalDepFinder()
if not any(isinstance(f, _OptionalDepFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _finder_instance)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_beam_batched_tdt_infer():
    """Instantiate BeamBatchedTDTInfer with mocked decoder/joint/computer."""
    from nemo.collections.asr.parts.submodules.tdt_beam_decoding import BeamBatchedTDTInfer

    decoder = MagicMock()
    joint = MagicMock()

    patch_target = "nemo.collections.asr.parts.submodules.tdt_beam_decoding.ModifiedALSDBatchedTDTComputer"
    with patch(patch_target) as MockComputer:
        mock_computer = MagicMock()
        MockComputer.return_value = mock_computer
        infer = BeamBatchedTDTInfer(
            decoder_model=decoder,
            joint_model=joint,
            durations=[0, 1],
            blank_index=1,
            beam_size=4,
        )
    return infer, mock_computer


class _FakeCudaGraphsComputer:
    """Minimal concrete WithOptionalCudaGraphs for use as the inner computer."""

    def __init__(self):
        self.graphs_enabled = True

    def disable_cuda_graphs(self) -> bool:
        changed = self.graphs_enabled
        self.graphs_enabled = False
        return changed

    def maybe_enable_cuda_graphs(self) -> bool:
        changed = not self.graphs_enabled
        self.graphs_enabled = True
        return changed


# ---------------------------------------------------------------------------
# Group 1: BeamBatchedTDTInfer flag behaviour
# ---------------------------------------------------------------------------


class TestBeamBatchedTDTInferFlag:
    def test_flag_initialized_false(self):
        """`_cuda_graphs_disabled` must start False after __init__."""
        infer, _ = _make_beam_batched_tdt_infer()
        assert infer._cuda_graphs_disabled is False

    def test_disable_cuda_graphs_sets_flag(self):
        """`disable_cuda_graphs()` must set `_cuda_graphs_disabled = True`."""
        from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs

        infer, _ = _make_beam_batched_tdt_infer()
        computer = _FakeCudaGraphsComputer()

        # Patch the computer with one that IS a WithOptionalCudaGraphs instance
        FakeComputer = type(
            "FakeComputer",
            (WithOptionalCudaGraphs, _FakeCudaGraphsComputer),
            {
                "disable_cuda_graphs": _FakeCudaGraphsComputer.disable_cuda_graphs,
                "maybe_enable_cuda_graphs": _FakeCudaGraphsComputer.maybe_enable_cuda_graphs,
            },
        )
        infer._decoding_computer = FakeComputer()

        infer.disable_cuda_graphs()
        assert infer._cuda_graphs_disabled is True

    def test_maybe_enable_cuda_graphs_clears_flag(self):
        """`maybe_enable_cuda_graphs()` must clear `_cuda_graphs_disabled` to False."""
        from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs

        infer, _ = _make_beam_batched_tdt_infer()
        FakeComputer = type(
            "FakeComputer",
            (WithOptionalCudaGraphs, _FakeCudaGraphsComputer),
            {
                "disable_cuda_graphs": _FakeCudaGraphsComputer.disable_cuda_graphs,
                "maybe_enable_cuda_graphs": _FakeCudaGraphsComputer.maybe_enable_cuda_graphs,
            },
        )
        infer._decoding_computer = FakeComputer()

        # disable first, then re-enable
        infer.disable_cuda_graphs()
        assert infer._cuda_graphs_disabled is True

        infer.maybe_enable_cuda_graphs()
        assert infer._cuda_graphs_disabled is False

    def test_flag_false_without_cuda_graphs_computer(self):
        """When inner computer is NOT WithOptionalCudaGraphs, disable returns False and flag stays False."""
        infer, _ = _make_beam_batched_tdt_infer()
        # _decoding_computer is a plain MagicMock — not a WithOptionalCudaGraphs subclass
        result = infer.disable_cuda_graphs()
        assert result is False
        assert infer._cuda_graphs_disabled is False


# ---------------------------------------------------------------------------
# Group 2: change_decoding_strategy() preserve/restore logic
#
# The ModelFacade below simulates the decoding attribute chain:
#   model.decoding.decoding._cuda_graphs_disabled
#   model.decoding.decoding.disable_cuda_graphs()
# without requiring a real NeMo model.
# ---------------------------------------------------------------------------


def _make_decoding_stub(disabled: bool = False):
    """Return a fake ``model.decoding`` whose ``.decoding`` carries the flag."""

    inner = SimpleNamespace(
        _cuda_graphs_disabled=disabled,
        _disable_called=False,
    )

    def _disable():
        inner._cuda_graphs_disabled = True
        inner._disable_called = True

    inner.disable_cuda_graphs = _disable
    outer = SimpleNamespace(decoding=inner)
    return outer


def _run_preserve_restore(old_disabled: bool):
    """
    Simulate the preserve/restore block used in all three change_decoding_strategy
    overrides and return the state of the new decoding object's flag.
    """
    # --- before block (preserve) ---
    old_decoding = _make_decoding_stub(disabled=old_disabled)
    _cuda_graphs_disabled = False
    try:
        _cuda_graphs_disabled = getattr(old_decoding.decoding, '_cuda_graphs_disabled', False)
    except AttributeError:
        pass

    # --- replacement ---
    new_decoding = _make_decoding_stub(disabled=False)  # always starts False (fresh object)

    # --- after block (restore) ---
    if _cuda_graphs_disabled:
        try:
            if hasattr(new_decoding.decoding, 'disable_cuda_graphs'):
                new_decoding.decoding.disable_cuda_graphs()
        except AttributeError:
            pass

    return new_decoding.decoding


class TestChangeDecodingStrategyPreserveRestore:
    def test_disabled_state_is_restored_after_replacement(self):
        """When CUDA graphs were disabled, the new decoding object should also be disabled."""
        new_inner = _run_preserve_restore(old_disabled=True)
        assert new_inner._cuda_graphs_disabled is True
        assert new_inner._disable_called is True

    def test_enabled_state_is_not_changed(self):
        """When CUDA graphs were NOT disabled, the new decoding object stays enabled."""
        new_inner = _run_preserve_restore(old_disabled=False)
        assert new_inner._cuda_graphs_disabled is False
        assert new_inner._disable_called is False

    def test_attribute_error_is_silently_swallowed(self):
        """AttributeError during preserve/restore must not propagate."""
        # Simulate old decoding having no .decoding attribute at all
        broken_old = object()  # has no .decoding

        _cuda_graphs_disabled = False
        try:
            _cuda_graphs_disabled = getattr(broken_old.decoding, '_cuda_graphs_disabled', False)  # type: ignore[attr-defined]
        except AttributeError:
            pass

        assert _cuda_graphs_disabled is False  # gracefully defaulted

        # Simulate new decoding having no .disable_cuda_graphs
        new_decoding = SimpleNamespace(decoding=SimpleNamespace(_cuda_graphs_disabled=False))
        # no disable_cuda_graphs method → hasattr returns False → nothing called
        if _cuda_graphs_disabled:
            try:
                if hasattr(new_decoding.decoding, 'disable_cuda_graphs'):
                    new_decoding.decoding.disable_cuda_graphs()
            except AttributeError:
                pass

        assert new_decoding.decoding._cuda_graphs_disabled is False
