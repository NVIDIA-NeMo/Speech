# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.inference.factory.base_builder import BaseBuilder
from nemo.collections.asr.inference.model_wrappers.cache_aware_ctc_inference_wrapper import (
    CacheAwareCTCInferenceWrapper,
)
from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
    CacheAwareRNNTInferenceWrapper,
)
from nemo.collections.asr.inference.model_wrappers.ctc_inference_wrapper import CTCInferenceWrapper
from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper

BUFFERED = [CTCInferenceWrapper, RNNTInferenceWrapper]
CACHE_AWARE = [CacheAwareCTCInferenceWrapper, CacheAwareRNNTInferenceWrapper]


class _Encoder:
    def forward_internal(self, audio_signal, length, bypass_pre_encode=False, **kwargs):
        return ("eager", audio_signal, length, bypass_pre_encode, kwargs)


def _build(wrapper_class, events, **kwargs):
    """Runs the real ASRInferenceWrapper.__init__ of the class with a fake model and a recording post init."""

    class _Wrapper(wrapper_class):
        @staticmethod
        def load_model(model_name, map_location):
            events.append("load_model")
            return SimpleNamespace(_cfg={}, tokenizer=None, encoder=_Encoder())

        def set_dither_to_zero(self):
            pass

        def __post_init__(self):
            events.append("post_init")

    _Wrapper.__abstractmethods__ = frozenset()
    return _Wrapper(model_name="m", decoding_cfg=None, device="cpu", compute_dtype="float32", **kwargs)


EVENTS = []


@pytest.fixture
def compile_calls(monkeypatch):
    calls = []
    EVENTS.clear()

    def fake_compile(target, **kwargs):
        EVENTS.append("compile")
        calls.append({"target": target, "kwargs": kwargs, "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR")})
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    return calls


@pytest.mark.unit
@pytest.mark.parametrize("wrapper_class", BUFFERED)
def test_buffered_wrappers_compile_the_whole_encoder_after_the_post_init(compile_calls, tmp_path, wrapper_class):
    wrapper = _build(wrapper_class, EVENTS, compile_encoder=True, compile_cache_dir=str(tmp_path / "cache"))

    assert EVENTS == ["load_model", "post_init", "compile"]
    assert len(compile_calls) == 1
    assert compile_calls[0]["target"] is wrapper.asr_model.encoder
    assert compile_calls[0]["cache_dir"] == str(tmp_path / "cache")  # already set when the compile starts
    assert os.path.isdir(tmp_path / "cache")


@pytest.mark.unit
@pytest.mark.parametrize("wrapper_class", CACHE_AWARE)
def test_cache_aware_wrappers_compile_the_encoder_body_with_dynamic_batch(compile_calls, tmp_path, wrapper_class):
    wrapper = _build(wrapper_class, EVENTS, compile_encoder=True, compile_cache_dir=str(tmp_path / "cache"))

    assert EVENTS == ["load_model", "post_init", "compile"]
    assert len(compile_calls) == 1
    assert compile_calls[0]["target"].__func__ is _Encoder.forward_internal  # the body, not the whole encoder
    assert compile_calls[0]["kwargs"] == {"dynamic": True}
    assert compile_calls[0]["cache_dir"] == str(tmp_path / "cache")
    assert isinstance(wrapper.asr_model.encoder, _Encoder)


@pytest.mark.unit
@pytest.mark.parametrize("wrapper_class", BUFFERED + CACHE_AWARE)
def test_nothing_is_compiled_or_configured_unless_requested(compile_calls, wrapper_class):
    wrapper = _build(wrapper_class, [], compile_encoder=False)

    assert compile_calls == []
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
    assert isinstance(wrapper.asr_model.encoder, _Encoder)


@pytest.mark.unit
@pytest.mark.parametrize("wrapper_class", BUFFERED + CACHE_AWARE)
def test_cache_dir_without_compile_is_an_error_before_the_model_is_loaded(compile_calls, tmp_path, wrapper_class):
    events = []
    with pytest.raises(ValueError, match="compile_cache_dir"):
        _build(wrapper_class, events, compile_encoder=False, compile_cache_dir=str(tmp_path))

    assert events == []


@pytest.mark.unit
def test_compiled_cache_aware_body_pins_the_feature_length_and_passes_the_cache_through(monkeypatch, compile_calls):
    pinned = []
    monkeypatch.setattr(torch._dynamo, "mark_static", lambda tensor, dim: pinned.append((tensor, dim)))
    wrapper = _build(CacheAwareCTCInferenceWrapper, [], compile_encoder=True)
    audio, embeddings = torch.zeros(3, 80, 17), torch.zeros(3, 17, 64)
    length, cache = torch.tensor([17, 17, 17]), torch.ones(2)

    out = wrapper.asr_model.encoder.forward_internal(audio, length, cache_last_time=cache)
    wrapper.asr_model.encoder.forward_internal(embeddings, length, bypass_pre_encode=True)

    assert out[0] == "eager" and out[1] is audio and out[2] is length
    assert out[3] is False and out[4] == {"cache_last_time": cache}
    # only the call that still runs the pre-encoder carries features, so only that one is pinned, on the time axis
    assert [(t is audio, dim) for t, dim in pinned] == [(True, 2)]


@pytest.mark.unit
@pytest.mark.parametrize("wrapper_class", CACHE_AWARE)
def test_cache_aware_encoder_without_a_body_cannot_be_compiled(compile_calls, wrapper_class):
    class _Wrapper(wrapper_class):
        @staticmethod
        def load_model(model_name, map_location):
            return SimpleNamespace(_cfg={}, tokenizer=None, encoder=object())

        def set_dither_to_zero(self):
            pass

        def __post_init__(self):
            pass

    _Wrapper.__abstractmethods__ = frozenset()
    with pytest.raises(ValueError, match="forward_internal"):
        _Wrapper(model_name="m", decoding_cfg=None, device="cpu", compute_dtype="float32", compile_encoder=True)


@pytest.mark.unit
@pytest.mark.parametrize("wrapper_class", CACHE_AWARE)
def test_compile_and_encoder_cuda_graphs_cannot_be_combined(compile_calls, wrapper_class):
    wrapper = _build(wrapper_class, [], compile_encoder=True)
    graph_calls = []
    wrapper.asr_model.encoder.set_streaming_cuda_graphs = lambda enabled: graph_calls.append(enabled)

    with pytest.raises(ValueError, match="cannot be combined"):
        wrapper.set_streaming_cuda_graphs(enabled=True)
    wrapper.set_streaming_cuda_graphs(enabled=False)

    assert graph_calls == [False]


def _builder_cfg(decoding_type, pipeline_type, **asr_cfg):
    asr = {"model_name": "m", "device": "cpu", "device_id": 0, "compute_dtype": "float32", "use_amp": False}
    return OmegaConf.create(
        {"asr_decoding_type": decoding_type, "pipeline_type": pipeline_type, "asr": {**asr, **asr_cfg}}
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "decoding_type, pipeline_type, wrapper_name",
    [
        ("ctc", "buffered", "CTCInferenceWrapper"),
        ("rnnt", "buffered", "RNNTInferenceWrapper"),
        ("ctc", "cache_aware", "CacheAwareCTCInferenceWrapper"),
        ("rnnt", "cache_aware", "CacheAwareRNNTInferenceWrapper"),
    ],
)
def test_builder_passes_both_compile_options_to_every_wrapper(monkeypatch, decoding_type, pipeline_type, wrapper_name):
    seen = {}

    class _Wrapper:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(f"nemo.collections.asr.inference.factory.base_builder.{wrapper_name}", _Wrapper)
    cfg = _builder_cfg(decoding_type, pipeline_type, compile_encoder=True, compile_cache_dir="/some/dir")

    BaseBuilder._build_asr(cfg, decoding_cfg=None)

    assert seen["compile_encoder"] is True and seen["compile_cache_dir"] == "/some/dir"


@pytest.mark.unit
def test_builder_leaves_the_options_out_when_compile_is_off(monkeypatch):
    seen = {}

    class _Wrapper:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("nemo.collections.asr.inference.factory.base_builder.CTCInferenceWrapper", _Wrapper)

    BaseBuilder._build_asr(_builder_cfg("ctc", "buffered"), decoding_cfg=None)

    assert "compile_encoder" not in seen and "compile_cache_dir" not in seen


@pytest.mark.unit
def test_builder_rejects_compile_for_the_salm_pipeline():
    cfg = _builder_cfg("salm", "buffered", compile_encoder=True)

    with pytest.raises(ValueError, match="SALM"):
        BaseBuilder._build_asr(cfg, decoding_cfg=None)


_COMPILE_TWICE = """
import sys, torch
from nemo.collections.asr.inference.utils.pipeline_utils import set_compile_cache_dir
from torch._dynamo.utils import counters
set_compile_cache_dir(sys.argv[1])
torch.manual_seed(0)
module = torch.compile(torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU()).cuda())
module(torch.randn(8, 64, device="cuda"))
print(counters["inductor"]["fxgraph_cache_hit"], counters["inductor"]["fxgraph_cache_miss"])
"""


@pytest.mark.gpu
def test_second_process_reuses_the_compiled_artifacts_from_the_cache_dir(tmp_path):
    def run():
        out = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_COMPILE_TWICE), str(tmp_path / "cache")],
            capture_output=True,
            text=True,
            check=True,
        )
        return [int(v) for v in out.stdout.split()[-2:]]

    first_hits, first_misses = run()
    second_hits, second_misses = run()

    assert (first_hits, first_misses) == (0, 1)
    assert (second_hits, second_misses) == (1, 0)
