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

"""Offline vs. incremental inference parity tests for NemotronVoiceChat.

``test_parity_tiny_model`` checks offline-vs-incremental parity using a
tiny model with random weights (no checkpoint needed, requires only a GPU).

``test_parity_real_checkpoint`` does the same on a real exported checkpoint
and is skipped unless ``PARITY_CHECKPOINT_PATH`` is set::

    PARITY_CHECKPOINT_PATH=/path/to/exported/checkpoint
    PARITY_AUDIO_PATH=/path/to/test.wav  # optional, defaults to force_align_test.mp3
    PARITY_SPEAKER_NAME=<name>           # optional

Run from the NeMo repo root (use ``-s`` to see live progress)::

    # unit tests only
    CUDA_VISIBLE_DEVICES=0 pytest tests/collections/speechlm2/voicechat/test_nemotron_voicechat_pipeline_parity.py -v -s

    # include integration test
    PARITY_CHECKPOINT_PATH=... \\
        CUDA_VISIBLE_DEVICES=0 pytest tests/collections/speechlm2/voicechat/test_nemotron_voicechat_pipeline_parity.py -v -s
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
    FRAME_SIZE_SAMPLES,
    SAMPLE_RATE,
)
from nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline import StreamingS2SPipeline
from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
from nemo.utils import logging

_CONF_YAML = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml",
    )
)
_FORCE_ALIGN_AUDIO = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_data",
        "force_align_test.mp3",
    )
)
_MOCK_SYSTEM_PROMPT = "This is a mock prompt for the test"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare_tensors(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
) -> dict[str, Any]:
    """Prefix-aware comparison of two tensors (tokens or logits, any shape)."""
    if a is None or b is None:
        return {"match": None, "note": "one or both tensors missing"}
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    T = min(a.shape[1], b.shape[1])
    if T == 0:
        return {"prefix_len": 0, "match": True}
    ap, bp = a[:, :T], b[:, :T]
    diff = (ap - bp).abs()
    match = bool(diff.max() == 0)
    result: dict[str, Any] = {"prefix_len": T, "match": match, "max_abs_diff": float(diff.max())}
    if not match:
        reduce = tuple(i for i in range(diff.dim()) if i != 1)
        per_step = diff.amax(dim=reduce) if reduce else diff.squeeze()
        nonzero = (per_step > 0).nonzero(as_tuple=False)
        if nonzero.numel():
            result["first_diff_step"] = int(nonzero[0].item())
    return result


def _merge_incremental_debug_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-step debug dicts from the pipeline into a single dict."""
    if not steps:
        return {}
    all_text_logits = [s["text_logits"] for s in steps if s.get("text_logits") is not None]
    all_asr_logits = [s["asr_logits"] for s in steps if s.get("asr_logits") is not None]
    return {
        "text_logits": torch.cat(all_text_logits, dim=1) if all_text_logits else None,
        "asr_logits": torch.cat(all_asr_logits, dim=1) if all_asr_logits else None,
    }


def _load_and_pad_audio(
    audio_path: str, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load audio, zero-pad to a whole number of 80 ms frames, return ``(audio, lens)``."""
    import librosa

    audio_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    padded_len = math.ceil(len(audio_np) / FRAME_SIZE_SAMPLES) * FRAME_SIZE_SAMPLES
    audio = torch.nn.functional.pad(
        torch.tensor(audio_np, device=device, dtype=dtype).unsqueeze(0),
        (0, max(0, padded_len - len(audio_np))),
    )
    return audio, torch.tensor([audio.shape[1]], device=device, dtype=torch.long)


def run_parity_check(
    pipeline: StreamingS2SPipeline,
    audio_path: str,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run offline and incremental inference on the same audio, return comparison.

    Only STT-level tokens and logits are compared; TTS is irrelevant for
    the core parity invariant.
    """
    wrapper = pipeline.s2s_model
    audio, audio_lens = _load_and_pad_audio(audio_path, wrapper.device, wrapper.dtype)

    prompt_tokens = prompt_token_lens = None
    if system_prompt:
        tok = wrapper.tokenizer
        ids = [tok.bos_id] + tok.text_to_ids(system_prompt) + [tok.eos_id]
        prompt_tokens = torch.tensor(ids, device=wrapper.device, dtype=torch.long).unsqueeze(0)
        prompt_token_lens = torch.tensor([len(ids)], device=wrapper.device, dtype=torch.long)

    if wrapper.speaker_name is not None:
        OmegaConf.update(wrapper.model.cfg, "inference_speaker_name", wrapper.speaker_name, force_add=True)
    elif wrapper.speaker_reference:
        OmegaConf.update(wrapper.model.cfg, "inference_speaker_reference", wrapper.speaker_reference, force_add=True)
    speaker_kw: dict[str, Any] = {}
    if not wrapper.model.cfg.get("inference_speaker_name") and not wrapper.model.cfg.get("inference_speaker_reference"):
        speaker_kw["speaker_audio"] = torch.randn(1, 22050, device=wrapper.device)
        speaker_kw["speaker_audio_lens"] = torch.tensor([22050], device=wrapper.device, dtype=torch.long)

    # -- Offline --
    logging.info("Running offline_inference ...")
    t0 = time.time()
    offline = wrapper.model.offline_inference(
        input_signal=audio,
        input_signal_lens=audio_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
        decode_audio=False,
        return_logits=True,
        **speaker_kw,
    )
    logging.info(f"  offline done in {time.time() - t0:.2f}s")

    # -- Incremental --
    logging.info("Running incremental inference (pipeline.run) ...")
    t0 = time.time()
    pipeline.collect_debug = True
    outputs = pipeline.run(
        [audio_path],
        options=[S2SRequestOptions(system_prompt=system_prompt)],
    )
    logging.info(f"  incremental done in {time.time() - t0:.2f}s")

    output = outputs[0]
    inc_tokens = output.token_text
    inc_asr_tokens = output.token_asr_text
    assert output.debug_data, "collect_debug=True but no debug data was recorded"
    inc_debug = _merge_incremental_debug_steps(output.debug_data)

    # -- Compare --
    # offline_inference returns logits for ALL positions (including prompt),
    # while the incremental path only produces logits for audio positions.
    # Trim the prompt prefix from offline logits so the two are aligned.
    prompt_len = prompt_tokens.shape[1] if prompt_tokens is not None else 0

    report: dict[str, Any] = {
        "token_comparison": _compare_tensors(offline.get("tokens_text"), inc_tokens),
        "asr_token_comparison": _compare_tensors(offline.get("tokens_text_src"), inc_asr_tokens),
    }
    for key, off_key, inc_key in [
        ("text_logit_comparison", "text_logits", "text_logits"),
        ("asr_logit_comparison", "asr_logits", "asr_logits"),
    ]:
        off_t, inc_t = offline.get(off_key), inc_debug.get(inc_key)
        if off_t is not None and prompt_len > 0:
            off_t = off_t[:, prompt_len:]
        if off_t is not None and inc_t is not None:
            report[key] = _compare_tensors(off_t, inc_t)

    return report


def assert_parity(
    report: dict[str, Any],
    *,
    strict: bool = True,
    atol: float = 0.0,
) -> None:
    """Raise ``AssertionError`` if parity checks in *report* fail."""
    failures: list[str] = []
    for key in ("token_comparison", "asr_token_comparison"):
        c = report.get(key, {})
        if c.get("match") is False:
            failures.append(f"{key}: diverge at step {c.get('first_diff_step')}")
    if strict:
        for key in ("text_logit_comparison", "asr_logit_comparison"):
            c = report.get(key, {})
            if c.get("match") is False and c.get("max_abs_diff", 0) > atol:
                failures.append(f"{key}: max_abs_diff={c['max_abs_diff']:.2e} > atol={atol:.2e}")
    assert not failures, "Parity failed:\n  " + "\n  ".join(failures)


# Parity requires deterministic, float32, and greedy decoding.
_PARITY_DEFAULTS = {
    "s2s": {
        "engine_type": "native",
        "compute_dtype": "float32",
        "deterministic": True,
        "decode_audio": False,
        "use_perception_cache": False,
        "use_perception_cudagraph": False,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
    },
}


def _build_parity_pipeline(
    model_path: str,
    audio_path: str,
    output_dir: str,
    *overrides: dict[str, Any],
) -> StreamingS2SPipeline:
    """Build a :class:`StreamingS2SPipeline` configured for strict parity testing.

    Loads ``s2s_streaming.yaml`` as the base config, applies
    ``_PARITY_DEFAULTS``, then merges each dict in *overrides*
    on top (in order).

    The chunk size is set to cover the full audio in one step so that
    offline and incremental paths see identical input.
    """
    import librosa

    audio_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    total_frames = math.ceil(len(audio_np) / FRAME_SIZE_SAMPLES)
    chunk_secs = total_frames * FRAME_SIZE_SAMPLES / SAMPLE_RATE

    cfg = OmegaConf.load(_CONF_YAML)
    cfg = OmegaConf.merge(
        cfg,
        _PARITY_DEFAULTS,
        {
            "audio_file": audio_path,
            "output_dir": output_dir,
            "s2s": {"model_path": model_path},
            "streaming": {
                "chunk_size_in_secs": chunk_secs,
                "buffer_size_in_secs": max(71 * 0.08, chunk_secs),
            },
        },
    )
    for ov in overrides:
        if ov:
            cfg = OmegaConf.merge(cfg, ov)
    return S2SPipelineBuilder.build_pipeline(cfg)


# ---------------------------------------------------------------------------
# Parity test -- tiny model (uses tiny_model_artifacts from conftest.py)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_parity_tiny_model(tiny_model_artifacts):
    """Offline/incremental parity with a tiny random-weight model.

    Loads the model through the real ``S2SPipelineBuilder`` so the test
    exercises the same code path as ``test_parity_real_checkpoint``.
    """
    model_dir, audio_path, _ = tiny_model_artifacts
    output_dir = tempfile.mkdtemp(prefix="parity-tiny-")

    pipeline = _build_parity_pipeline(
        model_dir, audio_path, output_dir,
        {"s2s": {"system_prompt": _MOCK_SYSTEM_PROMPT}},
    )
    report = run_parity_check(pipeline, audio_path, system_prompt=_MOCK_SYSTEM_PROMPT)
    assert_parity(report, strict=True, atol=0.0)


# ---------------------------------------------------------------------------
# Integration test -- real checkpoint (skipped when env vars are not set)
# ---------------------------------------------------------------------------


def _real_checkpoint_available() -> bool:
    path = os.environ.get("PARITY_CHECKPOINT_PATH", "")
    return bool(path) and os.path.isdir(path)


@pytest.mark.skipif(not _real_checkpoint_available(), reason="set PARITY_CHECKPOINT_PATH")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_parity_real_checkpoint():
    """Parity check using a real exported checkpoint.

    Configure via environment variables::

        PARITY_CHECKPOINT_PATH=/path/to/exported/checkpoint
        PARITY_AUDIO_PATH=/path/to/test.wav  # optional, defaults to force_align_test.mp3
        PARITY_SPEAKER_NAME=<name>           # optional
    """
    ckpt = os.environ["PARITY_CHECKPOINT_PATH"]
    audio = os.environ.get("PARITY_AUDIO_PATH") or _FORCE_ALIGN_AUDIO
    speaker = os.environ.get("PARITY_SPEAKER_NAME")

    parity_overrides: dict[str, Any] = {"s2s": {"system_prompt": _MOCK_SYSTEM_PROMPT}}
    if speaker:
        parity_overrides["s2s"]["speaker_name"] = speaker
    pipeline = _build_parity_pipeline(
        ckpt, audio, tempfile.mkdtemp(prefix="parity-"),
        parity_overrides,
    )
    report = run_parity_check(pipeline, audio, system_prompt=_MOCK_SYSTEM_PROMPT)
    assert_parity(report, strict=True, atol=0.0)
