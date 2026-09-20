# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Export a streaming STT checkpoint to HuggingFace Hub format.

A streaming-STT front-end over :mod:`to_hf`, which it reuses for everything that is not
specific to this recipe (config rendering, DTensor consolidation, distributed setup, vLLM
prep). Keeping it a separate entry point means ``to_hf.py`` stays byte-identical to upstream
and never conflicts on a merge.

It differs from ``to_hf.py`` in exactly two ways:

* ``weights_only`` is threaded into :func:`torch.load`, so a checkpoint carrying pickled
  objects can still be read with ``weights_only=false``.
* ``prepare_vllm`` defaults to **False**. Streaming checkpoints are consumed by
  ``StreamingSTTModel`` / ``StreamingSTTModelAutomodel``, not served through vLLM, and the
  vLLM prep step rewrites ``config.json`` and drops a tokenizer and generation config next
  to it. Pass ``prepare_vllm=true`` to get the upstream behavior.

Examples:
    # Single-file checkpoint:
    python streaming_stt_to_hf.py \\
        class_path=nemo.collections.speechlm2.models.StreamingSTTModel \\
        ckpt_path=/path/to/checkpoint.ckpt \\
        ckpt_config=/path/to/config.yaml \\
        output_dir=/path/to/hf_output

    # Distributed (FSDP2/TP) checkpoint — same GPU count as training:
    torchrun --nproc-per-node=8 streaming_stt_to_hf.py \\
        class_path=nemo.collections.speechlm2.models.StreamingSTTModelAutomodel \\
        ckpt_path=/path/to/distributed_ckpt_dir \\
        ckpt_config=/path/to/config.yaml \\
        output_dir=/path/to/hf_output
"""

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from nemo.collections.speechlm2.parts.hf_hub import LLM_BACKBONE_DIR
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.dtype import str_to_dtype
from nemo.utils.model_utils import import_class_by_path

# ``to_hf`` is a sibling script rather than an installed module, so make the import
# independent of the launcher's working directory (plain ``python``, ``torchrun``, ``srun``).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from to_hf import (  # noqa: E402
    HfExportConfig,
    _canonical_torch_dtype_name,
    _hf_export_config,
    _try_prepare_for_vllm,
    _uses_automodel_parallel,
    consolidate_state_dict,
    save_hf_checkpoint,
    save_llm_backbone_config,
    setup_distributed_from_config,
)


@dataclass
class StreamingSTTHfExportConfig(HfExportConfig):
    """``HfExportConfig`` plus the two knobs the streaming recipe needs."""

    # Forwarded to torch.load. Set false to read a checkpoint holding pickled objects.
    weights_only: bool = True

    # Write the llm_backbone config and the vLLM-ready artifacts (patched config.json,
    # tokenizer, generation_config). Off by default: streaming checkpoints are consumed by
    # StreamingSTTModel, and the non-automodel export path already covers the serving case.
    prepare_vllm: bool = False


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, weights_only: bool = True) -> None:
    """Load a Lightning checkpoint, honoring ``weights_only`` for the single-file case."""
    if Path(checkpoint_path).is_dir():
        from torch.distributed.checkpoint import load

        state_dict = {"state_dict": model.state_dict()}
        load(state_dict, checkpoint_id=checkpoint_path)
        model.load_state_dict(state_dict["state_dict"])
    else:
        ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
        model.load_state_dict(ckpt_data["state_dict"])


@hydra_runner(config_name="StreamingSTTHfExportConfig", schema=StreamingSTTHfExportConfig)
def main(cfg: StreamingSTTHfExportConfig) -> None:
    """Read a PyTorch Lightning checkpoint and export it to HuggingFace Hub format.

    The result is loadable via ``ModelClass.from_pretrained(path)``. Distributed checkpoints
    written by ``AutomodelParallelStrategy`` are supported; parallelism sizes are read from
    the ``trainer.strategy`` section of ``ckpt_config``, so launch under ``torchrun`` with the
    same number of GPUs used for training.
    """
    if not Path(cfg.ckpt_path).exists():
        raise RuntimeError(f"No such file or directory: {cfg.ckpt_path}")

    full_cfg = OmegaConf.to_container(OmegaConf.load(cfg.ckpt_config), resolve=True)
    model_cfg = full_cfg["model"]
    audio_token_estimator = full_cfg.get("data", {}).get("train_ds", {}).get("audio_token_estimator")
    if audio_token_estimator is not None:
        # The vLLM prompt processor must reserve exactly as many audio
        # placeholders as the checkpoint's encoder emits.
        model_cfg["audio_token_estimator"] = audio_token_estimator
    model_cfg["torch_dtype"] = _canonical_torch_dtype_name(cfg.dtype)
    cls = import_class_by_path(cfg.class_path)

    strategy_cfg = full_cfg.get("trainer", {}).get("strategy", {})

    _is_torchrun = "RANK" in os.environ
    if _is_torchrun and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    is_distributed = (
        _is_torchrun
        and Path(cfg.ckpt_path).is_dir()
        and _uses_automodel_parallel(strategy_cfg)
        and dist.get_world_size() > 1
    )

    if is_distributed:
        strategy = setup_distributed_from_config(strategy_cfg)

        # Don't call configure_model() inside __init__ — we set the distributed setup first.
        model_cfg["init_configure_model"] = False
        model_cfg["pretrained_weights"] = False
        model = cls(model_cfg)
        model.configure_model(distributed_setup=strategy.distributed_setup)

        load_checkpoint(model, cfg.ckpt_path, weights_only=cfg.weights_only)

        # Consolidate DTensors to regular tensors and save on rank 0.
        consolidated = consolidate_state_dict(model)
        if dist.get_rank() == 0:
            # Upstream's save_hf_checkpoint always writes the llm_backbone config; drop it
            # again rather than reimplementing the function, so the dtype/state-dict-adapter
            # handling stays shared with to_hf.py instead of drifting from it.
            save_hf_checkpoint(model, consolidated, cfg)
            if cfg.prepare_vllm:
                save_llm_backbone_config(model, cfg.output_dir)
                _try_prepare_for_vllm(cfg.output_dir, model_cfg)
            else:
                shutil.rmtree(Path(cfg.output_dir) / LLM_BACKBONE_DIR, ignore_errors=True)

        dist.barrier()
        dist.destroy_process_group()
    else:
        model_cfg["init_configure_model"] = True
        model_cfg["pretrained_weights"] = False
        model = cls(model_cfg)
        load_checkpoint(model, cfg.ckpt_path, weights_only=cfg.weights_only)
        model = model.to(str_to_dtype(cfg.dtype))
        # save_pretrained() writes no llm_backbone config, so this path adds it when asked
        # instead of removing it.
        model.save_pretrained(cfg.output_dir, config=_hf_export_config(model, cfg.dtype))
        if cfg.prepare_vllm:
            save_llm_backbone_config(model, cfg.output_dir)
            _try_prepare_for_vllm(cfg.output_dir, model_cfg)

    logging.info(f"Model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
