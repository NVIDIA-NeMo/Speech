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
from unittest.mock import Mock

import pytest
import torch

from examples.asr.asr_cache_aware_streaming import speech_to_text_cache_aware_streaming_infer as infer


@pytest.mark.unit
@pytest.mark.parametrize("compare_vs_offline", [False, True])
@pytest.mark.parametrize("debug_mode", [False, True])
def test_single_audio_file_returns_before_dataset_metrics(monkeypatch, compare_vs_offline, debug_mode):
    cfg = infer.TranscriptionConfig(
        audio_file="audio.wav",
        cuda=-1,
        compare_vs_offline=compare_vs_offline,
        debug_mode=debug_mode,
        pad_and_drop_preencoded=True,
    )
    model = SimpleNamespace(encoder=SimpleNamespace(streaming_cfg={}), to=Mock(), eval=Mock())
    model.to.return_value = model
    buffer = SimpleNamespace(append_audio_file=Mock())
    expected = (["hello world"], ["hello world"] if compare_vs_offline else None)
    perform_streaming = Mock(return_value=expected)
    word_error_rate = Mock(side_effect=AssertionError("Single-file input must skip dataset metrics"))

    monkeypatch.setattr(infer, "setup_model", Mock(return_value=(model, "model")))
    monkeypatch.setattr(infer, "CacheAwareStreamingAudioBuffer", Mock(return_value=buffer))
    monkeypatch.setattr(infer, "get_inference_device", Mock(return_value=torch.device("cpu")))
    monkeypatch.setattr(infer, "get_inference_dtype", Mock(return_value=torch.float32))
    monkeypatch.setattr(infer, "perform_streaming", perform_streaming)
    monkeypatch.setattr(infer, "word_error_rate", word_error_rate)
    # Keep the process-wide torch settings unchanged for other tests.
    monkeypatch.setattr(torch, "set_grad_enabled", Mock())
    monkeypatch.setattr(torch, "set_float32_matmul_precision", Mock())

    assert infer.main(cfg) == expected

    buffer.append_audio_file.assert_called_once_with("audio.wav", stream_id=-1)
    perform_streaming.assert_called_once_with(
        asr_model=model,
        streaming_buffer=buffer,
        compute_dtype=torch.float32,
        compare_vs_offline=compare_vs_offline,
        debug_mode=debug_mode,
        pad_and_drop_preencoded=True,
    )
    word_error_rate.assert_not_called()
