# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Functional checks for training, resuming and exporting a tiny Transcribe model."""

import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader, load
from transformers import AutoTokenizer

from tests.functional_tests.nemotron35_transcribe_fixture import (
    RESUME_STEP,
    SPEAKER_COUNT,
    TRAINING_STEPS,
    prepare_training_inputs,
)


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("train")


@pytest.fixture(scope="module")
def continuous(work_dir):
    cfg = prepare_training_inputs(work_dir, Path(os.environ.get("NEMOTRON35_TEST_DATA", "/home/TestData")))
    config = work_dir / "continuous.yaml"
    OmegaConf.save(cfg, config)
    _torchrun("-m", "tests.functional_tests.nemotron35_transcribe_fixture", str(config))
    return torch.load(work_dir / "continuous.pt", weights_only=True)


@pytest.fixture(scope="module")
def resumed(work_dir, continuous):
    cfg = OmegaConf.load(work_dir / "continuous.yaml")
    cfg.exp_manager.explicit_log_dir = str(work_dir / "resumed")
    cfg.exp_manager.resume_if_exists = True
    cfg.exp_manager.resume_from_checkpoint = str(work_dir / "continuous/checkpoints" / f"step={RESUME_STEP}.ckpt")
    config = work_dir / "resumed.yaml"
    OmegaConf.save(cfg, config)
    _torchrun("-m", "tests.functional_tests.nemotron35_transcribe_fixture", str(config))
    return torch.load(work_dir / "resumed.pt", weights_only=True)


@pytest.fixture(scope="module")
def export_dir(work_dir, resumed):
    output = work_dir / "export"
    checkpoint = work_dir / "resumed/checkpoints" / f"step={TRAINING_STEPS}.ckpt"
    _torchrun(
        "examples/speechlm2/to_hf.py",
        "class_path=nemo.collections.speechlm2.models.SALMAutomodel",
        f"ckpt_path='{checkpoint}'",
        f"ckpt_config={work_dir}/resumed/exp_config.yaml",
        f"output_dir={output}",
    )
    return output


@pytest.fixture(scope="module")
def export_config(export_dir):
    return json.loads((export_dir / "config.json").read_text())


@pytest.fixture(scope="module")
def tokenizer(work_dir, continuous):
    return AutoTokenizer.from_pretrained(work_dir / "fixture/model", local_files_only=True)


def test_training_batches_pack_text_and_waveforms(continuous):
    for step in continuous["steps"]:
        batch = step["batch"]
        assert batch["input_ids"].ndim == 1
        assert batch["text_cu_seqlens"].numel() > 2
        assert batch["text_cu_seqlens"][-1] == batch["input_ids"].numel()
        assert batch["packed_audio_samples"].shape == (int(batch["audio_lens"].sum()),)


@pytest.mark.parametrize("answer", ["contextual", "earlier", "later"])
def test_selected_context_answer_and_both_assistant_turns_are_supervised(continuous, tokenizer, answer):
    token_id = tokenizer.convert_tokens_to_ids(answer)
    assert token_id != tokenizer.unk_token_id
    found = False
    for step in continuous["steps"]:
        batch = step["batch"]
        positions = batch["input_ids"] == token_id
        if positions.any():
            found = True
            assert batch["loss_mask"][positions].all()
    assert found, f"Training never saw the expected answer token {answer!r}"


def test_audio_placeholders_are_not_supervised(continuous, tokenizer):
    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")
    for step in continuous["steps"]:
        batch = step["batch"]
        positions = batch["input_ids"] == audio_token_id
        assert positions.any()
        assert not batch["loss_mask"][positions].any()


def test_audio_without_rttm_requests_sortformer_predictions(continuous, work_dir):
    rows = _speaker_targets(continuous, sf.info(work_dir / "fixture/short.wav").frames)
    assert rows, "No short audio examples reached the training step"
    for targets in rows:
        assert targets.shape[-1] == SPEAKER_COUNT
        assert (targets == -1).all()


def test_rttm_keeps_all_eight_overlapping_speaker_tracks(continuous, work_dir):
    rows = _speaker_targets(continuous, sf.info(work_dir / "fixture/long.wav").frames)
    assert rows, "No long audio examples reached the training step"
    for targets in rows:
        assert targets.shape[-1] == SPEAKER_COUNT
        assert (targets == 1).any(dim=0).all()
        assert (targets == 1).all(dim=1).any()


def test_training_updates_audio_projection(continuous):
    assert continuous["projection_changed"]


def test_sortformer_stays_frozen_during_training(continuous):
    assert continuous["diarization_frozen"]


def test_mel_frontend_leaves_normalization_to_the_encoder(continuous):
    assert continuous["mel_normalization"] is None


@pytest.mark.parametrize("block", ["asr", "mtp"])
def test_trainable_blocks_receive_finite_nonzero_gradients(continuous, block):
    gradients = [step["max_gradients"][block] for step in continuous["steps"]]
    assert all(math.isfinite(value) for value in gradients)
    assert any(value > 0 for value in gradients)


@pytest.mark.parametrize("loss", ["loss", "mtp_loss", "mtp_loss_unscaled/head_1", "mtp_loss_unscaled/head_2"])
def test_training_and_both_mtp_depths_report_finite_positive_losses(continuous, loss):
    for step in continuous["steps"]:
        value = step["losses"][loss]
        assert math.isfinite(value)
        assert value > 0


def test_resume_restores_fused_adam_state_and_step(continuous, resumed):
    assert continuous["start_step"] == 0
    assert resumed["start_step"] == RESUME_STEP
    assert resumed["optimizer"] == continuous["optimizer"] == "FusedAdam"
    saved = continuous["adam_at_resume_step"]
    restored = resumed["adam_at_resume_step"]
    assert restored["group_steps"] == saved["group_steps"]
    assert RESUME_STEP in saved["group_steps"]
    assert any(moment["exp_avg"].count_nonzero() for moment in saved["moments"].values())
    # Resume can allocate zero buffers for parameters without checkpointed moments.
    for parameter, moments in saved["moments"].items():
        torch.testing.assert_close(restored["moments"][parameter], moments, rtol=0, atol=0)


def test_resume_replays_packed_text_audio_speaker_targets_and_lr(continuous, resumed):
    original_steps = continuous["steps"]
    replayed_steps = resumed["steps"]
    assert [step["step"] for step in original_steps] == list(range(1, TRAINING_STEPS + 1))
    for original, replay in zip(original_steps[RESUME_STEP:], replayed_steps, strict=True):
        assert original["step"] == replay["step"]
        assert original["lr"] == replay["lr"]
        torch.testing.assert_close(original["batch"], replay["batch"], rtol=0, atol=0)


def test_export_has_portable_backbone_and_inline_encoder(export_config):
    assert export_config["architectures"] == ["Nemotron3_5TranscribeForConditionalGeneration"]
    assert export_config["pretrained_llm"] == "llm_backbone"
    assert export_config["pe_encoder_path"] is None
    encoder = export_config["pe_encoder_config"]
    assert encoder["asr_normalize_type"] == "per_feature"
    assert encoder["diar_normalize_type"] == "NA"
    assert encoder["chunk_size_seconds"] == 30
    assert encoder["speaker_feature_mode"] == "thresholded"
    assert encoder["speaker_activity_threshold"] == 0.5


def test_export_keeps_two_repeated_mtp_depths(export_config):
    assert export_config["mtp"]["num_nextn_predict_layers"] == 2
    assert export_config["mtp"]["use_repeated_layer"]


@pytest.mark.parametrize(
    "prefix", ["perception.encoder.asr_encoder.", "perception.encoder.diarization_model.", "llm.mtp."]
)
def test_export_includes_both_encoder_branches_and_mtp_weights(export_dir, prefix):
    with safe_open(export_dir / "model.safetensors", framework="pt") as checkpoint:
        assert any(name.startswith(prefix) for name in checkpoint.keys())


@pytest.mark.parametrize("name", ["A_log", "e_score_correction_bias"])
def test_bf16_export_preserves_exact_fp32_checkpoint_weights(export_dir, work_dir, name):
    with safe_open(export_dir / "model.safetensors", framework="pt") as exported:
        keys = [key for key in exported.keys() if key.endswith(f".{name}")]
        assert keys, f"Missing exported {name}"
        reader = FileSystemReader(work_dir / "resumed/checkpoints" / f"step={TRAINING_STEPS}.ckpt")
        metadata = reader.read_metadata().state_dict_metadata
        original = {}
        for key in keys:
            tensor = metadata[f"state_dict.{key}"]
            assert tensor.properties.dtype == torch.float32
            original[key] = torch.empty(tensor.size, dtype=torch.float32)
        load({"state_dict": original}, storage_reader=reader)
        for key in keys:
            torch.testing.assert_close(exported.get_tensor(key), original[key], rtol=0, atol=0)


def _speaker_targets(run, samples):
    rows = []
    for step in run["steps"]:
        batch = step["batch"]
        offsets = batch["spk_target_cu_seqlens"].tolist()
        lengths = batch["audio_lens"].tolist()
        for length, start, end in zip(lengths, offsets[:-1], offsets[1:], strict=True):
            if length == samples:
                rows.append(batch["spk_targets"][start:end])
    return rows


def _torchrun(*args):
    env = os.environ.copy()
    # SM120 uses Transformer Engine's FlashAttention path for packed backward.
    if torch.cuda.get_device_capability() == (12, 0):
        env["NVTE_FUSED_ATTN"] = "0"
    subprocess.run(
        [
            "torchrun",
            "--nproc-per-node=1",
            "--rdzv-backend=static",
            "--master-addr=127.0.0.1",
            "--local-addr=127.0.0.1",
            "--no-python",
            "coverage",
            "run",
            "--parallel-mode",
            "--source=nemo",
            *args,
        ],
        env=env,
        check=True,
    )
