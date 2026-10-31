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

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint


_CHECKPOINT_AVERAGING_PATH = (
    Path(__file__).parents[3]
    / "examples"
    / "speaker_tasks"
    / "diarization"
    / "neural_diarizer"
    / "checkpoint_averaging.py"
)
_SPEC = importlib.util.spec_from_file_location("checkpoint_averaging_for_test", _CHECKPOINT_AVERAGING_PATH)
_CHECKPOINT_AVERAGING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKPOINT_AVERAGING)
average_checkpoints_after_training = _CHECKPOINT_AVERAGING.average_checkpoints_after_training


class _FakeStrategy:
    def __init__(self):
        self.barrier_calls = []

    def barrier(self, name):
        self.barrier_calls.append(name)


class _FakeTrainer:
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.is_global_zero = True
        self.strategy = _FakeStrategy()


class _FakeModel:
    def __init__(self):
        self.loaded_state = None
        self.load_strict = None
        self.saved_paths = []

    def load_state_dict(self, state_dict, strict):
        self.loaded_state = {key: value.clone() for key, value in state_dict.items()}
        self.load_strict = strict

    def save_to(self, path):
        self.saved_paths.append(path)
        Path(path).write_bytes(b"saved")


@pytest.mark.unit
def test_average_checkpoints_after_training_averages_and_saves(tmp_path):
    lower_ranked_path = tmp_path / "lower.ckpt"
    higher_ranked_path = tmp_path / "higher.ckpt"
    torch.save(
        {"state_dict": {"weight": torch.tensor([1.0, 3.0]), "step": torch.tensor(99, dtype=torch.int64)}},
        lower_ranked_path,
    )
    torch.save(
        {"state_dict": {"weight": torch.tensor([3.0, 5.0]), "step": torch.tensor(7, dtype=torch.int64)}},
        higher_ranked_path,
    )

    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, monitor="score", mode="max", save_top_k=2)
    checkpoint_callback.best_k_models = {
        str(lower_ranked_path): torch.tensor(0.1),
        str(higher_ranked_path): torch.tensor(0.9),
    }
    trainer = _FakeTrainer(callbacks=[checkpoint_callback])
    model = _FakeModel()

    result = average_checkpoints_after_training(trainer=trainer, model=model)

    output_path = tmp_path / "model_averaged.nemo"
    assert result == str(output_path)
    assert output_path.exists()
    assert model.saved_paths == [str(output_path)]
    assert model.load_strict is True
    assert torch.equal(model.loaded_state["weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(model.loaded_state["step"], torch.tensor(7, dtype=torch.int64))
    assert trainer.strategy.barrier_calls == ["checkpoint_averaging_done"]


@pytest.mark.unit
@pytest.mark.parametrize("with_empty_callback", [False, True], ids=["missing-callback", "empty-best-k"])
def test_average_checkpoints_after_training_skips_when_no_checkpoints(tmp_path, with_empty_callback):
    callbacks = []
    if with_empty_callback:
        checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, monitor="score", mode="max", save_top_k=2)
        checkpoint_callback.best_k_models = {}
        callbacks.append(checkpoint_callback)

    trainer = _FakeTrainer(callbacks=callbacks)
    model = Mock()

    result = average_checkpoints_after_training(trainer=trainer, model=model)

    assert result == ""
    model.load_state_dict.assert_not_called()
    model.save_to.assert_not_called()
    assert trainer.strategy.barrier_calls == ["checkpoint_averaging_done"]
