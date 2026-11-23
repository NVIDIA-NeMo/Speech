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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.inference.model_wrappers.asr_inference_wrapper import ASRInferenceWrapper
from nemo.collections.asr.inference.model_wrappers.cache_aware_ctc_inference_wrapper import (
    CacheAwareCTCInferenceWrapper,
)
from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
    CacheAwareRNNTInferenceWrapper,
)
from nemo.collections.asr.inference.model_wrappers.ctc_inference_wrapper import CTCInferenceWrapper
from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

VOCABULARY = ["<unk>", "▁a", "b", "c"]


class _Wrapper(ASRInferenceWrapper):
    """Wrapper with the model-specific setup stubbed out, to exercise the shared construction."""

    decoder_type = 'rnnt'

    def __post_init__(self):
        self.post_init_done = True

    def get_vocabulary(self) -> list[str]:
        return self.asr_model.joint.vocabulary


@pytest.mark.unit
@pytest.mark.parametrize("decoding_cfg", [OmegaConf.structured(RNNTDecodingConfig), CTCDecodingConfig()])
def test_regular_model_passes_the_guard(monkeypatch, decoding_cfg):
    wrapper = _make_wrapper(monkeypatch, _Wrapper, _fake_model(VOCABULARY), decoding_cfg)
    assert wrapper.post_init_done


@pytest.mark.unit
@pytest.mark.parametrize(
    "wrapper_cls",
    [
        _Wrapper,
        RNNTInferenceWrapper,
        CTCInferenceWrapper,
        CacheAwareRNNTInferenceWrapper,
        CacheAwareCTCInferenceWrapper,
    ],
)
@pytest.mark.parametrize("eou_tokens", [["<EOU>", "<EOB>"], ["<EOU>"], ["<EOB>"]])
def test_end_of_utterance_model_raises(monkeypatch, wrapper_cls, eou_tokens):
    """EOU models stop emitting after the first <EOU>; the pipelines must not silently drop the rest of the audio."""
    monkeypatch.setattr(wrapper_cls, "__post_init__", lambda self: None)  # model-specific setup needs a real model
    model = _fake_model(VOCABULARY + eou_tokens)
    with pytest.raises(NotImplementedError, match="End-of-utterance models .* not supported .*restart_after_tokens"):
        _make_wrapper(monkeypatch, wrapper_cls, model, OmegaConf.structured(RNNTDecodingConfig))


@pytest.mark.unit
def test_model_type_is_checked_before_the_vocabulary(monkeypatch):
    model = _fake_model(VOCABULARY + ["<EOU>"])
    with pytest.raises(ValueError, match="not a CTC type"):
        _make_wrapper(monkeypatch, CacheAwareCTCInferenceWrapper, model, CTCDecodingConfig())


@pytest.mark.unit
def test_pipeline_restart_after_tokens_raises(monkeypatch):
    decoding_cfg = OmegaConf.structured(RNNTDecodingConfig(restart_after_tokens=["c"]))
    with pytest.raises(NotImplementedError, match="`decoding.restart_after_tokens` is not supported"):
        _make_wrapper(monkeypatch, _Wrapper, _fake_model(VOCABULARY), decoding_cfg)


@pytest.mark.unit
@pytest.mark.parametrize("wrapper_cls", [_Wrapper, RNNTInferenceWrapper, CacheAwareRNNTInferenceWrapper])
def test_model_restart_after_tokens_raises_for_rnnt(monkeypatch, wrapper_cls):
    monkeypatch.setattr(wrapper_cls, "__post_init__", lambda self: None)
    model = _fake_model(VOCABULARY, model_decoding_cfg={"restart_after_tokens": ["c"]})
    with pytest.raises(NotImplementedError, match="`decoding.restart_after_tokens` is not supported"):
        _make_wrapper(monkeypatch, wrapper_cls, model, OmegaConf.structured(RNNTDecodingConfig))


@pytest.mark.unit
@pytest.mark.parametrize("wrapper_cls", [CTCInferenceWrapper, CacheAwareCTCInferenceWrapper])
def test_hybrid_model_rnnt_restart_after_tokens_is_ignored_with_ctc(monkeypatch, wrapper_cls):
    """The model `decoding` config of a hybrid model belongs to its RNNT decoder, which CTC pipelines do not use."""
    monkeypatch.setattr(wrapper_cls, "__post_init__", lambda self: None)
    model = _fake_model(VOCABULARY, model_decoding_cfg={"restart_after_tokens": ["c"]})
    assert isinstance(_make_wrapper(monkeypatch, wrapper_cls, model, CTCDecodingConfig()), wrapper_cls)


@pytest.mark.unit
def test_error_points_to_the_cli_option_of_the_cache_aware_streaming_script(monkeypatch):
    with pytest.raises(NotImplementedError, match="streaming_infer.py` with `rnnt_decoding.restart_after_tokens`"):
        _make_wrapper(monkeypatch, _Wrapper, _fake_model(VOCABULARY + ["<EOU>"]), CTCDecodingConfig())


def _fake_model(vocabulary, model_decoding_cfg=None):
    return SimpleNamespace(
        _cfg=OmegaConf.create({"preprocessor": {"dither": 1e-5}, "decoding": model_decoding_cfg or {}}),
        tokenizer=object(),
        preprocessor=SimpleNamespace(featurizer=SimpleNamespace(dither=1e-5)),
        joint=SimpleNamespace(vocabulary=vocabulary),
        decoder=SimpleNamespace(vocabulary=vocabulary),
        ctc_decoder=SimpleNamespace(vocabulary=vocabulary),
    )


def _make_wrapper(monkeypatch, wrapper_cls, model, decoding_cfg):
    monkeypatch.setattr(ASRInferenceWrapper, "load_model", staticmethod(lambda model_name, map_location: model))
    return wrapper_cls(model_name="fake", decoding_cfg=decoding_cfg, device="cpu", compute_dtype="float32")
