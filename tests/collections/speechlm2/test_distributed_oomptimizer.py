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
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import click
import lightning.pytorch as pl
import pytest
import torch
from click.testing import CliRunner
from lightning.pytorch.strategies import ModelParallelStrategy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy
from nemo.utils.trainer_utils import resolve_trainer_cfg
from tests.collections.speechlm2.distributed_oomptimizer_model import SingleBlockDistributedOOMptimizerModel

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "distributed_oomptimizer_for_test", _REPO_ROOT / "scripts/speechlm2/distributed_oomptimizer.py"
)
_OOMPTIMIZER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OOMPTIMIZER)


def test_single_process_trainer_cfg_keeps_recipe_strategy():
    # The Automodel strategy is what casts SALMAutomodel's perception module to the training dtype on a single GPU.
    # Dropping it profiles fp32 trainable parameters and breaks the recipe's FlashAdamW optimizer.
    cfg = OmegaConf.load(_REPO_ROOT / "examples/speechlm2/conf/salm_automodel.yaml")
    trainer_cfg = _OOMPTIMIZER._single_process_trainer_cfg(resolve_trainer_cfg(cfg.trainer))

    assert isinstance(trainer_cfg["strategy"], AutomodelParallelStrategy)
    assert trainer_cfg["devices"] == 1
    assert trainer_cfg["num_nodes"] == 1


def test_single_process_trainer_cfg_without_strategy():
    trainer_cfg = _OOMPTIMIZER._single_process_trainer_cfg(
        {"devices": 8, "num_nodes": 2, "accelerator": "gpu", "max_steps": 1}
    )

    assert "strategy" not in trainer_cfg
    assert trainer_cfg == {"devices": 1, "num_nodes": 1, "accelerator": "gpu", "max_steps": 1}


@pytest.mark.parametrize("strategy", ["ddp_spawn", "ddp_fork", "ddp_notebook"])
def test_rejects_strategies_with_spawning_launchers(strategy):
    trainer = _make_trainer(accelerator="cpu", strategy=strategy)

    with pytest.raises(click.ClickException, match="Use 'ddp'"):
        _OOMPTIMIZER._validate_in_process_launcher(trainer)


@pytest.mark.parametrize("strategy", ["auto", "ddp"])
def test_accepts_in_process_strategies(strategy):
    _OOMPTIMIZER._validate_in_process_launcher(_make_trainer(accelerator="cpu", strategy=strategy))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_entrypoint_rejects_ddp_spawn_before_building_model(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        {
            "model": {"vocab_size": 64, "sample_rate": 32, "frame_stride": 4, "hidden_size": 16, "num_heads": 2},
            "trainer": {"devices": 1, "accelerator": "gpu", "strategy": "ddp_spawn", "logger": False},
        },
        config_path,
    )
    build_model = Mock(side_effect=AssertionError("the model must not be built"))
    monkeypatch.setattr(SingleBlockDistributedOOMptimizerModel, "__init__", build_model)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    try:
        result = CliRunner().invoke(
            _OOMPTIMIZER.oomptimizer,
            [
                "-c",
                str(config_path),
                "-m",
                f"{SingleBlockDistributedOOMptimizerModel.__module__}.SingleBlockDistributedOOMptimizerModel",
                "-b",
                "[12.0]",
                "--no-distributed-supervisor",
            ],
        )
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)

    assert result.exit_code != 0
    assert "Use 'ddp'" in result.output
    build_model.assert_not_called()


def test_release_warmup_optimizers_clears_state():
    module = _TinyModule()
    trainer = _make_trainer(accelerator="cpu", strategy="auto")
    trainer.fit(module, DataLoader(_TinyDataset(), batch_size=None))
    (warmup_optimizer,) = trainer.optimizers
    assert len(warmup_optimizer.state) > 0

    _OOMPTIMIZER._release_warmup_optimizers(trainer)

    assert len(warmup_optimizer.state) == 0
    # Models may still read the learning rate through the trainer during profiling.
    assert trainer.optimizers[0].param_groups[0]["lr"] == 1e-3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_release_warmup_optimizers_frees_model_parallel_optimizer_state_on_gpu():
    # ModelParallelStrategy.teardown does not offload optimizer state to CPU (unlike the generic Strategy.teardown),
    # so without the release the warm-up optimizer's state would stay on the GPU while profiling allocates a second one.
    module = _TinyModule(configure_model_hook=True)
    trainer = _make_trainer(accelerator="gpu", strategy=ModelParallelStrategy())
    try:
        trainer.fit(module, DataLoader(_TinyDataset(), batch_size=None))
        (warmup_optimizer,) = trainer.optimizers
        state_tensors = [t for state in warmup_optimizer.state.values() for t in state.values() if torch.is_tensor(t)]
        state_bytes = sum(t.numel() * t.element_size() for t in state_tensors if t.is_cuda)
        assert state_bytes > 0, "expected the warm-up optimizer state to remain on the GPU after fit"
        del state_tensors
        weight_device = module.proj.weight.device
        allocated_before = torch.cuda.memory_allocated()

        _OOMPTIMIZER._release_warmup_optimizers(trainer)

        assert len(warmup_optimizer.state) == 0
        assert torch.cuda.memory_allocated() <= allocated_before - state_bytes
        assert module.proj.weight.device == weight_device

        # The profiling optimizer gets the only set of optimizer state.
        profiling_optimizer = module.configure_optimizers()
        module.proj(torch.randn(4, 64, device=weight_device)).sum().backward()
        profiling_optimizer.step()
        assert len(profiling_optimizer.state) > 0
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class _TinyDataset(IterableDataset):
    def __iter__(self):
        yield torch.randn(4, 64)

    def __len__(self):
        return 1


class _TinyModule(pl.LightningModule):
    def __init__(self, configure_model_hook: bool = False):
        super().__init__()
        self.proj = torch.nn.Linear(64, 64)
        if configure_model_hook:
            # ModelParallelStrategy requires the hook to be overridden.
            self.configure_model = lambda: None

    def training_step(self, batch, batch_idx):
        return self.proj(batch.to(self.device)).sum()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def _make_trainer(accelerator: str, strategy) -> pl.Trainer:
    return pl.Trainer(
        accelerator=accelerator,
        devices=1,
        strategy=strategy,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
