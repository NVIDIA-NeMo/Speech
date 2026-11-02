# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a single LR-only change after restoring a full training checkpoint."""

from lightning.pytorch.callbacks import Callback


class ScaleRestoredLearningRate(Callback):
    """Scale the restored cosine schedule once, preserving all other state.

    The callback state is checkpointed so later segments of the new experiment
    resume without applying the scale a second time. Keep the callback configuration
    unchanged on subsequent resumes. The first full-state restore must use a
    canonical ``/step=<source_step>.ckpt`` path; see the SpeechLM training guide's
    "Reducing LR After a Full-State Resume" section for configuration examples.
    """

    def __init__(self, factor: float, source_step: int, original_base_lr: float, original_min_lr: float):
        super().__init__()
        if not 0.0 < factor < 1.0:
            raise ValueError(f"Expected an LR reduction factor in (0, 1), got {factor}")
        self.factor = float(factor)
        self.source_step = int(source_step)
        self.original_base_lr = float(original_base_lr)
        self.original_min_lr = float(original_min_lr)
        self.applied = False

    @property
    def state_key(self) -> str:
        return f"{type(self).__qualname__}(factor={self.factor},source_step={self.source_step})"

    def state_dict(self) -> dict:
        return {"applied": self.applied}

    def load_state_dict(self, state_dict: dict) -> None:
        self.applied = bool(state_dict["applied"])

    def on_train_start(self, trainer, pl_module) -> None:
        if self.applied:
            if trainer.global_step <= self.source_step:
                raise RuntimeError("LR scale marked applied before the source step")
            return

        if trainer.global_step != self.source_step:
            raise RuntimeError(f"LR-only restart expected step {self.source_step}, got {trainer.global_step}")
        checkpoint = str(trainer.ckpt_path)
        if not checkpoint.endswith(f"/step={self.source_step}.ckpt"):
            raise RuntimeError(f"LR-only restart expected the canonical source checkpoint, got {checkpoint}")
        if len(trainer.optimizers) != 1 or len(trainer.lr_scheduler_configs) != 1:
            raise RuntimeError("LR-only restart requires exactly one restored optimizer and scheduler")

        optimizer = trainer.optimizers[0]
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        if type(scheduler).__name__ != "CosineAnnealing":
            raise RuntimeError(f"Unexpected restored LR scheduler: {type(scheduler).__name__}")
        if len(optimizer.param_groups) != len(scheduler.base_lrs):
            raise RuntimeError("Restored optimizer and scheduler parameter groups differ")
        if any(abs(float(lr) - self.original_base_lr) > 1e-12 for lr in scheduler.base_lrs):
            raise RuntimeError(f"Unexpected restored base LRs: {scheduler.base_lrs}")
        if abs(float(scheduler.min_lr) - self.original_min_lr) > 1e-12:
            raise RuntimeError(f"Unexpected restored minimum LR: {scheduler.min_lr}")

        old_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * self.factor
            if "initial_lr" in group:
                group["initial_lr"] = float(group["initial_lr"]) * self.factor
        scheduler.base_lrs = [float(lr) * self.factor for lr in scheduler.base_lrs]
        scheduler.min_lr = float(scheduler.min_lr) * self.factor
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [float(lr) * self.factor for lr in scheduler._last_lr]
        self.applied = True
        if trainer.is_global_zero:
            print(
                f"LR_ONLY_RESUME source_step={self.source_step} factor={self.factor} "
                f"old_lrs={old_lrs} new_lrs={[group['lr'] for group in optimizer.param_groups]} "
                f"base_lrs={scheduler.base_lrs} min_lr={scheduler.min_lr}",
                flush=True,
            )
