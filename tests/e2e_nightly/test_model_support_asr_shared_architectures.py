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
Model-support tests for released FastConformer and Parakeet ASR checkpoints that share an architecture.

Each checkpoint is restored, runs one training step on random data, and transcribes random audio. Checkpoints with a
dedicated test module (Canary, multitalker, Sortformer, and so on) are not listed here. The models are split into
`NEMO_MODEL_SUPPORT_NUM_SHARDS` CI jobs; `NEMO_MODEL_SUPPORT_SHARD` selects one of them.
"""

import gc
import os

import numpy as np
import pytest
import torch

# (model name, checkpoint file, decoder type)
ASR_MODELS = [
    ("nvidia/nemotron-speech-streaming-en-0.6b", "nvidia__nemotron-speech-streaming-en-0.6b.nemo", "rnnt"),
    ("nvidia/parakeet-rnnt-0.6b", "nvidia__parakeet-rnnt-0.6b.nemo", "rnnt"),
    ("nvidia/parakeet-rnnt-1.1b", "nvidia__parakeet-rnnt-1.1b.nemo", "rnnt"),
    ("nvidia/parakeet-rnnt-110m-da-dk", "nvidia__parakeet-rnnt-110m-da-dk.nemo", "rnnt"),
    (
        "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0",
        "nvidia__stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo",
        "rnnt",
    ),
    (
        "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
        "nvidia__stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo",
        "rnnt",
    ),
    ("nvidia/stt_be_fastconformer_hybrid_large_pc", "nvidia__stt_be_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_de_fastconformer_hybrid_large_pc", "nvidia__stt_de_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_en_fastconformer_hybrid_large_pc", "nvidia__stt_en_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    (
        "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
        "nvidia__stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
        "rnnt",
    ),
    (
        "nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms",
        "nvidia__stt_en_fastconformer_hybrid_medium_streaming_80ms.nemo",
        "rnnt",
    ),
    (
        "nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc",
        "nvidia__stt_en_fastconformer_hybrid_medium_streaming_80ms_pc.nemo",
        "rnnt",
    ),
    ("nvidia/stt_en_fastconformer_transducer_large", "nvidia__stt_en_fastconformer_transducer_large.nemo", "rnnt"),
    ("nvidia/stt_en_fastconformer_transducer_xlarge", "nvidia__stt_en_fastconformer_transducer_xlarge.nemo", "rnnt"),
    ("nvidia/stt_en_fastconformer_transducer_xxlarge", "nvidia__stt_en_fastconformer_transducer_xxlarge.nemo", "rnnt"),
    ("nvidia/stt_es_fastconformer_hybrid_large_pc", "nvidia__stt_es_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_es_fastconformer_hybrid_large_pc_nc", "nvidia__stt_es_fastconformer_hybrid_large_pc_nc.nemo", "rnnt"),
    ("nvidia/stt_fa_fastconformer_hybrid_large", "nvidia__stt_fa_fastconformer_hybrid_large.nemo", "rnnt"),
    ("nvidia/stt_fr_fastconformer_hybrid_large_pc", "nvidia__stt_fr_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_hr_fastconformer_hybrid_large_pc", "nvidia__stt_hr_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_hy_fastconformer_hybrid_large_pc", "nvidia__stt_hy_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_it_fastconformer_hybrid_large_pc", "nvidia__stt_it_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_ka_fastconformer_hybrid_large_pc", "nvidia__stt_ka_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    (
        "nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc",
        "nvidia__stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc.nemo",
        "rnnt",
    ),
    ("nvidia/stt_kk_ru_fastconformer_hybrid_large", "nvidia__stt_kk_ru_fastconformer_hybrid_large.nemo", "rnnt"),
    ("nvidia/stt_nl_fastconformer_hybrid_large_pc", "nvidia__stt_nl_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_pl_fastconformer_hybrid_large_pc", "nvidia__stt_pl_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_pt_fastconformer_hybrid_large_pc", "nvidia__stt_pt_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_ru_fastconformer_hybrid_large_pc", "nvidia__stt_ru_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_ua_fastconformer_hybrid_large_pc", "nvidia__stt_ua_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    ("nvidia/stt_uz_fastconformer_hybrid_large_pc", "nvidia__stt_uz_fastconformer_hybrid_large_pc.nemo", "rnnt"),
    (
        "stt_en_fastconformer_hybrid_large_streaming_1040ms",
        "stt_en_fastconformer_hybrid_large_streaming_1040ms.nemo",
        "rnnt",
    ),
    (
        "stt_multilingual_fastconformer_hybrid_large_pc_blend_eu",
        "stt_multilingual_fastconformer_hybrid_large_pc_blend_eu.nemo",
        "rnnt",
    ),
    ("nvidia/parakeet-tdt-0.6b-v2", "nvidia__parakeet-tdt-0.6b-v2.nemo", "tdt"),
    ("nvidia/parakeet-tdt-0.6b-v3", "nvidia__parakeet-tdt-0.6b-v3.nemo", "tdt"),
    ("nvidia/parakeet-tdt-1.1b", "nvidia__parakeet-tdt-1.1b.nemo", "tdt"),
    ("nvidia/parakeet-tdt_ctc-0.6b-ja", "nvidia__parakeet-tdt_ctc-0.6b-ja.nemo", "tdt"),
    ("nvidia/parakeet-tdt_ctc-1.1b", "nvidia__parakeet-tdt_ctc-1.1b.nemo", "tdt"),
    ("nvidia/parakeet-tdt_ctc-110m", "nvidia__parakeet-tdt_ctc-110m.nemo", "tdt"),
    ("nvidia/stt_en_fastconformer_tdt_large", "nvidia__stt_en_fastconformer_tdt_large.nemo", "tdt"),
    ("nvidia/parakeet-ctc-0.6b", "nvidia__parakeet-ctc-0.6b.nemo", "ctc"),
    ("nvidia/parakeet-ctc-0.6b-Vietnamese", "nvidia__parakeet-ctc-0.6b-Vietnamese.nemo", "ctc"),
    ("nvidia/parakeet-ctc-1.1b", "nvidia__parakeet-ctc-1.1b.nemo", "ctc"),
    ("nvidia/stt_en_fastconformer_ctc_large", "nvidia__stt_en_fastconformer_ctc_large.nemo", "ctc"),
    ("nvidia/stt_en_fastconformer_ctc_xlarge", "nvidia__stt_en_fastconformer_ctc_xlarge.nemo", "ctc"),
    ("nvidia/stt_en_fastconformer_ctc_xxlarge", "nvidia__stt_en_fastconformer_ctc_xxlarge.nemo", "ctc"),
]

MODEL_DIR = os.environ.get(
    "NEMO_MODEL_SUPPORT_DIR",
    os.environ.get("NEMO_MODEL_SUPPORT_DIR_CI", "/home/TestData/nemo-speech-ci-models"),
)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SHARD = int(os.environ.get("NEMO_MODEL_SUPPORT_SHARD", "0"))
_NUM_SHARDS = int(os.environ.get("NEMO_MODEL_SUPPORT_NUM_SHARDS", "1"))


@pytest.fixture(
    scope="module",
    params=ASR_MODELS[_SHARD::_NUM_SHARDS],
    ids=lambda params: params[0],
)
def model_and_decoder_type(request):
    from nemo.collections.asr.models import ASRModel

    _, nemo_file, decoder_type = request.param
    # Free the previous checkpoint before loading the next one. Collect here rather than at teardown: pytest still
    # holds the previous parameter's yielded value during teardown, and models contain reference cycles.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = ASRModel.restore_from(os.path.join(MODEL_DIR, nemo_file), map_location="cpu").to(_DEVICE)
    yield model, decoder_type


def test_model_init(model_and_decoder_type):
    model, _ = model_and_decoder_type
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


def test_model_training_step(model_and_decoder_type):
    """Run one training step via direct training_step() call."""
    from e2e_utils import prepare_for_training_step

    model, decoder_type = model_and_decoder_type
    prepare_for_training_step(model)
    d = next(model.parameters()).device
    vocab_size = _vocab_size(model, decoder_type)
    batch = (
        torch.randn(2, 16000, device=d),
        torch.tensor([16000, 12000], device=d),
        torch.randint(0, max(1, vocab_size), (2, 5), dtype=torch.long, device=d),
        torch.tensor([5, 3], dtype=torch.long, device=d),
    )
    result = model.training_step(batch, 0)
    loss = result if isinstance(result, torch.Tensor) else result['loss']
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()


def test_model_inference(model_and_decoder_type):
    """Test full inference pipeline via model.transcribe()."""
    model, _ = model_and_decoder_type
    model.eval()

    audio = np.random.randn(16000).astype(np.float32)

    result = model.transcribe(audio=[audio], batch_size=1)
    assert isinstance(result, list)
    assert len(result) == 1
    # transcribe() may return strings or Hypothesis objects
    text = result[0] if isinstance(result[0], str) else result[0].text
    assert isinstance(text, str)

    hyps = model.transcribe(audio=[audio], batch_size=1, return_hypotheses=True)
    assert isinstance(hyps, list)
    assert len(hyps) == 1
    assert hasattr(hyps[0], 'text')


def _vocab_size(model, decoder_type: str) -> int:
    """Number of transcript token IDs to sample random training targets from."""
    if decoder_type == "ctc":
        return model.tokenizer.vocab_size
    if decoder_type == "tdt":
        # TDT joints may include duration outputs; transcript targets must use only decoder token IDs.
        return model.decoder.vocab_size
    return model.joint.num_classes_with_blank - 1
