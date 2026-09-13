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
# pylint: disable=C0116
"""
Training-throughput metrics that are not specific to a single model.

Three metrics are emitted at ``on_train_batch_end`` via ``pl_module.log()``:

* ``dataloader_wait_s`` — wall-clock seconds spent between the previous
  batch's ``on_train_batch_end`` and the current batch's
  ``on_train_batch_start``. With PTL's prefetcher this is normally near
  zero; large values mean the dataloader couldn't keep up. Useful for
  catching AIS / lustre stalls before they crater the run.
* ``num_tokens_total`` — running sum across the whole training of every
  non-padding token position fed into the LLM (text non-pad + audio
  frames after perception subsampling). Includes loss-masked tokens.
  Survives job restarts via callback ``state_dict``.
* ``num_examples_total`` — running sum across the whole training of
  per-batch example counts. Also restart-safe.

The model is expected to populate two short-lived attributes inside its
``training_step`` so the callback can pick them up without parsing the
batch a second time::

    pl_module._last_batch_num_tokens = int(...)
    pl_module._last_batch_num_examples = int(...)

If either attribute is missing the callback falls back to counting from
``batch["input_ids"]`` (non-pad text tokens only) — useful for non-SALM
models, but loses audio-frame contribution.
"""

import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from lightning.pytorch import Callback, LightningModule, Trainer

__all__ = ["TrainingStatsCallback"]


class TrainingStatsCallback(Callback):
    """Logs dataloader wait time and accumulates token/example counts.

    Persists ``num_tokens_total`` and ``num_examples_total`` via the
    Lightning checkpoint state-dict mechanism so the counters survive
    job restarts. The per-step ``dataloader_wait_s`` gauge is
    intentionally NOT persisted (it has no meaningful value across a
    process boundary).

    The first batch after a fresh process start has no meaningful
    ``dataloader_wait_s`` (no preceding ``on_train_batch_end``); the
    callback skips logging it for that step.
    """

    def __init__(self) -> None:
        super().__init__()
        # Persisted state — survives checkpoint resume.
        self.num_tokens_total: int = 0
        self.num_examples_total: int = 0
        # Per-process state — not persisted.
        self._prev_batch_end_monotonic: Optional[float] = None

    # ------------------------------------------------------------------ state
    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_tokens_total": int(self.num_tokens_total),
            "num_examples_total": int(self.num_examples_total),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.num_tokens_total = int(state_dict.get("num_tokens_total", 0))
        self.num_examples_total = int(state_dict.get("num_examples_total", 0))

    # ----------------------------------------------------------------- hooks
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._prev_batch_end_monotonic is None:
            # First batch of the process — no previous end timestamp to
            # diff against; skip emitting a misleading value.
            return
        wait_s = time.monotonic() - self._prev_batch_end_monotonic
        # ``batch_size`` is required when the LightningModule uses
        # ``def training_step(self, dataloader_iter)`` (SALMAutomodel does):
        # Lightning can't auto-infer it from a ``dataloader_iter`` arg.
        # The value is only used for epoch-level aggregation; we log
        # ``on_step=True, on_epoch=False`` so the actual number is
        # irrelevant — pass 1 as a sentinel.
        pl_module.log(
            "dataloader_wait_s",
            wait_s,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            rank_zero_only=True,
            batch_size=1,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Pull per-batch counts the model exposed in training_step.
        local_tokens = int(getattr(pl_module, "_last_batch_num_tokens", -1))
        local_examples = int(getattr(pl_module, "_last_batch_num_examples", -1))
        if local_tokens < 0 or local_examples < 0:
            # Model didn't expose the attributes — fall back to a generic
            # estimate from the batch. Counts non-pad text tokens only;
            # audio frame contribution is lost. Better than zero.
            local_tokens, local_examples = self._fallback_counts(batch, pl_module)

        # All-reduce across DP ranks so every rank holds the same cumulative
        # value (required for state_dict consistency across ranks on save).
        # Under CP/TP, batch broadcasting gives model-parallel ranks duplicate
        # data, so reducing over the full world would over-count.
        if dist.is_available() and dist.is_initialized():
            buf = torch.tensor(
                [local_tokens, local_examples],
                dtype=torch.long,
                device=pl_module.device,
            )
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=self._get_dp_group(pl_module))
            global_tokens, global_examples = buf.tolist()
        else:
            global_tokens, global_examples = local_tokens, local_examples

        self.num_tokens_total += global_tokens
        self.num_examples_total += global_examples

        pl_module.log_dict(
            {
                "num_tokens_total": float(self.num_tokens_total),
                "num_examples_total": float(self.num_examples_total),
            },
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            rank_zero_only=True,
            batch_size=max(local_examples, 1),
        )

        self._prev_batch_end_monotonic = time.monotonic()

    # ------------------------------------------------------------ fallbacks
    @staticmethod
    def _fallback_counts(batch: Any, pl_module: LightningModule) -> tuple[int, int]:
        """Best-effort token/example count from a generic ``batch`` dict.

        Used only when the model didn't expose
        ``_last_batch_num_tokens`` / ``_last_batch_num_examples``. Counts
        non-pad text tokens via ``batch["input_ids"]`` and
        ``pl_module.text_pad_id`` when both exist. Audio-frame
        contribution is not visible from here.
        """
        try:
            ids = batch["input_ids"]
        except (KeyError, TypeError):
            return 0, 0
        if not torch.is_tensor(ids):
            return 0, 0
        pad_id = getattr(pl_module, "text_pad_id", None)
        if pad_id is None:
            n_tokens = int(ids.numel())
        else:
            n_tokens = int((ids != pad_id).long().sum().item())
        n_examples = int(ids.shape[0])
        return n_tokens, n_examples

    @staticmethod
    def _get_dp_group(pl_module: LightningModule):
        """Return a DP-only process group when model parallelism is active.

        ``None`` intentionally means the default world group, which is correct
        for plain DDP and single-process runs.
        """
        device_mesh = getattr(pl_module, "_device_mesh", None)
        if device_mesh is None:
            trainer = getattr(pl_module, "trainer", None)
            trainer_model = getattr(trainer, "model", None)
            device_mesh = getattr(trainer_model, "device_mesh", None)
        if device_mesh is None:
            return None

        names = device_mesh.mesh_dim_names or ()
        if "data_parallel" in names:
            return device_mesh["data_parallel"].get_group()

        try:
            from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

            return get_flat_mesh(device_mesh, "dp").get_group()
        except (ImportError, KeyError, RuntimeError, ValueError):
            pass

        try:
            return device_mesh["dp"].get_group()
        except (KeyError, RuntimeError, ValueError):
            pass

        if "dp_shard" in names and "dp_replicate" in names:
            return device_mesh["dp_replicate", "dp_shard"].get_group()
        if "dp_shard" in names:
            return device_mesh["dp_shard"].get_group()
        return None
