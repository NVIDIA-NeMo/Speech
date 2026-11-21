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

import json
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
from examples.asr.asr_cache_aware_streaming import speech_to_text_cache_aware_streaming_infer as streaming
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import EncDecCTCModel


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_streaming_compute_dtype(tmp_path, monkeypatch, dtype):
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Requires native CUDA bfloat16 support")

    model = EncDecCTCModel(
        cfg=DictConfig(
            {
                "preprocessor": {
                    "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                    "normalize": "per_feature",
                },
                "encoder": {
                    "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                    "feat_in": 64,
                    "n_layers": 1,
                    "d_model": 16,
                    "n_heads": 2,
                    "att_context_size": [8, 1],
                    "att_context_style": "chunked_limited",
                    "conv_context_size": "causal",
                    "conv_kernel_size": 5,
                },
                "decoder": {
                    "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
                    "feat_in": 16,
                    "num_classes": 2,
                    "vocabulary": ["a", "b"],
                },
            }
        )
    )
    monkeypatch.setattr(streaming, "setup_model", lambda cfg, map_location: (model, "tiny"))
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.random.default_rng(42).normal(0, 0.1, 16000), 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": str(audio_path), "text": "a"}))
    cfg = OmegaConf.structured(
        streaming.TranscriptionConfig(
            cuda=0,
            compute_dtype=dtype,
            dataset_manifest=str(manifest),
            matmul_precision=torch.get_float32_matmul_precision(),
        )
    )
    with patch.object(model.encoder, "cache_aware_stream_step", wraps=model.encoder.cache_aware_stream_step) as step:
        with torch.no_grad():
            streaming.main(cfg)

    assert step.call_count > 1
    expected = getattr(torch, dtype)
    assert next(model.encoder.parameters()).dtype == expected
    for call in step.call_args_list:
        for key in ("processed_signal", "cache_last_channel", "cache_last_time"):
            assert call.kwargs[key].dtype == expected
        assert call.kwargs["cache_last_channel_len"].dtype == torch.int64
