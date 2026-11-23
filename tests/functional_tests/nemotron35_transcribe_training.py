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
"""Assertions for the real SALM training CLI and its resumed run."""

import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from lightning.pytorch import Callback
from omegaconf import OmegaConf
from safetensors import safe_open
from torch.distributed.tensor import DTensor

from tests.functional_tests.nemotron35_transcribe_fixture import (
    CHUNK_SECONDS,
    CONTEXT_ANSWER,
    EARLIER_ANSWER,
    FINAL_ANSWER,
    SAMPLE_RATE,
    SPEAKER_COUNT,
)


class TranscribeTrainingChecks(Callback):
    """Record each rank's actual packed inputs to compare after restoring a DCP."""

    def __init__(self, receipt_dir: str):
        super().__init__()
        self.receipt_dir = Path(receipt_dir)
        self.batches = []
        self.seen = set()
        self.gradient_blocks = set()

    def on_train_start(self, trainer, pl_module):
        self.initial_step = trainer.global_step
        self.initial_projection = _local_tensor(pl_module.perception.proj.weight).detach().clone()
        optimizer = trainer.optimizers[0]
        assert optimizer.__class__.__name__ == "FusedAdam"
        if self.initial_step:
            assert any("exp_avg" in state for state in optimizer.state.values()), "FusedAdam state was not restored"
        encoder = pl_module.perception.encoder
        assert pl_module.perception.preprocessor.featurizer.normalize is None
        assert encoder.asr_normalize_type == "per_feature"
        assert encoder.diar_normalize_type == "NA"
        assert encoder.speaker_feature_mode == "thresholded"
        assert encoder.speaker_activity_threshold == 0.5
        assert encoder.chunk_size_seconds == CHUNK_SECONDS
        assert encoder.n_spk == SPEAKER_COUNT
        assert all(not parameter.requires_grad for parameter in encoder.diarization_model.parameters())
        assert pl_module.llm.mtp_config.num_layers == 2
        assert pl_module.llm.mtp_config.use_repeated_layer
        self.answer_ids = {
            answer: pl_module.tokenizer.text_to_ids(answer)[0]
            for answer in (CONTEXT_ANSWER, EARLIER_ANSWER, FINAL_ANSWER)
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ids, mask = batch["input_ids"], batch["loss_mask"]
        assert ids.ndim == 1 and batch["text_cu_seqlens"].numel() > 2
        assert mask.any()
        assert not mask[ids == pl_module.audio_locator_tag_id].any()
        for answer, token_id in self.answer_ids.items():
            present = ids == token_id
            if present.any():
                assert mask[present].all(), f"Missing supervision for {answer}"
                self.seen.add(answer)

        lengths = batch["audio_lens"]
        if lengths.numel():
            assert batch["packed_audio_samples"].shape == (int(lengths.sum()),)
            activity = batch["spk_targets"]
            assert activity.shape[-1] == SPEAKER_COUNT
            for length, start, end in zip(
                lengths.tolist(), batch["spk_target_cu_seqlens"][:-1], batch["spk_target_cu_seqlens"][1:]
            ):
                targets = activity[start:end]
                if length > CHUNK_SECONDS * SAMPLE_RATE:
                    assert (targets == 1).any(dim=0).all(), "Lost one or more RTTM speaker columns"
                    assert (targets == 1).all(dim=1).any(), "Lost overlapped RTTM supervision"
                    self.seen.add("long with RTTM")
                else:
                    assert (targets == -1).all(), "Missing RTTM must request predicted speaker activity"
                    self.seen.add("short without RTTM")

        metrics = trainer.callback_metrics
        for name in ("loss", "mtp_loss", "mtp_loss_unscaled/head_1", "mtp_loss_unscaled/head_2"):
            assert torch.isfinite(metrics[name]) and metrics[name] > 0, f"Invalid {name}"
        digest = hashlib.sha256()
        for key in ("input_ids", "loss_mask", "text_cu_seqlens", "packed_audio_samples", "spk_targets"):
            if (tensor := batch.get(key)) is not None:
                digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        self.batches.append(
            {
                "step": trainer.global_step,
                "input_sha256": digest.hexdigest(),
                "lr": trainer.optimizers[0].param_groups[0]["lr"],
            }
        )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for name, block in (("ASR encoder", pl_module.perception.encoder.asr_encoder), ("MTP", pl_module.llm.mtp)):
            grads = [_local_tensor(parameter.grad) for parameter in block.parameters() if parameter.grad is not None]
            if grads:
                grad_maxima = torch.stack([grad.detach().float().abs().amax() for grad in grads])
                assert torch.isfinite(grad_maxima).all(), f"Non-finite {name} gradients"
                if grad_maxima.any():
                    self.gradient_blocks.add(name)

    def on_train_end(self, trainer, pl_module):
        projection = _local_tensor(pl_module.perception.proj.weight)
        assert not torch.equal(self.initial_projection, projection), "Training did not update the audio projection"
        assert self.gradient_blocks == {"ASR encoder", "MTP"}
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt = {"initial_step": self.initial_step, "batches": self.batches, "seen": sorted(self.seen)}
        (self.receipt_dir / f"rank-{trainer.global_rank}.json").write_text(json.dumps(receipt))


def run_training(work_dir: Path, resume: bool) -> None:
    # NeMo requires TE's FlashAttention dispatch for packed backward on SM120.
    if torch.cuda.get_device_capability(int(os.environ["LOCAL_RANK"])) == (12, 0):
        os.environ["NVTE_FUSED_ATTN"] = "0"
    from examples.speechlm2 import salm_train

    cfg = OmegaConf.load(work_dir / "train.yaml")
    receipt_dir = work_dir / "receipts"
    if resume:
        cfg.exp_manager.explicit_log_dir = str(work_dir / "resumed")
        cfg.exp_manager.resume_from_checkpoint = str(work_dir / "train" / "checkpoints" / "step=2.ckpt")
        cfg.exp_manager.resume_if_exists = True
        receipt_dir = work_dir / "resumed_receipts"
    # Attach the assertion callback as a Python object while keeping the regular training entry point.
    with patch.object(
        salm_train, "Trainer", partial(salm_train.Trainer, callbacks=[TranscribeTrainingChecks(receipt_dir)])
    ):
        salm_train.train(cfg)


def verify_training_and_export(work_dir: Path, world_size: int) -> None:
    seen = set()
    for rank in range(world_size):
        original = json.loads((work_dir / "receipts" / f"rank-{rank}.json").read_text())
        resumed = json.loads((work_dir / "resumed_receipts" / f"rank-{rank}.json").read_text())
        assert original["initial_step"] == 0 and resumed["initial_step"] == 2
        assert [batch["step"] for batch in original["batches"]] == [1, 2, 3, 4]
        assert resumed["batches"] == original["batches"][2:], f"Rank {rank} did not replay the next packed inputs"
        seen.update(original["seen"])
    assert seen == {CONTEXT_ANSWER, EARLIER_ANSWER, FINAL_ANSWER, "long with RTTM", "short without RTTM"}

    directory = work_dir / "export"
    config = json.loads((directory / "config.json").read_text())
    assert config["architectures"] == ["Nemotron3_5TranscribeForConditionalGeneration"]
    assert config["pretrained_llm"] == "llm_backbone"
    assert config["pe_encoder_path"] is None
    encoder = config["pe_encoder_config"]
    assert encoder["asr_normalize_type"] == "per_feature" and encoder["diar_normalize_type"] == "NA"
    assert encoder["chunk_size_seconds"] == CHUNK_SECONDS and encoder["speaker_feature_mode"] == "thresholded"
    assert config["mtp"]["num_nextn_predict_layers"] == 2 and config["mtp"]["use_repeated_layer"]
    with safe_open(directory / "model.safetensors", framework="pt") as checkpoint:
        keys = checkpoint.keys()
        assert any(key.startswith("perception.encoder.asr_encoder.") for key in keys)
        assert any(key.startswith("perception.encoder.diarization_model.") for key in keys)
        assert any(key.startswith("llm.mtp.") for key in keys)
        for name in ("A_log", "e_score_correction_bias"):
            protected = [key for key in keys if key.endswith(f".{name}")]
            assert protected, f"Missing protected export tensor {name}"
            assert all(checkpoint.get_slice(key).get_dtype() == "F32" for key in protected), f"Lost FP32 for {name}"
    print(f"Transcribe training, exact sampler replay on {world_size} rank(s), and vLLM-ready export passed")


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("phase", choices=("train", "resume", "verify"))
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "verify":
        verify_training_and_export(args.work_dir, args.world_size)
    else:
        run_training(args.work_dir, resume=args.phase == "resume")
