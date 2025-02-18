# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock

import pytest
import torch

from nemo.lightning.io.hf import HFCheckpointIO


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.save_pretrained = MagicMock()
    model.load_pretrained = MagicMock(return_value={'mock_state_dict': torch.tensor([1.0])})
    return model


@pytest.fixture
def checkpoint_io(mock_model, tmp_path):
    return HFCheckpointIO(model=mock_model, adapter_only=False)


@pytest.fixture
def adapter_checkpoint_io(mock_model, tmp_path):
    return HFCheckpointIO(model=mock_model, adapter_only=True)


def save_and_load_checkpoint(checkpoint_io, checkpoint, path, adapter_only=False):
    try:
        if adapter_only:
            adapter_path = path / "hf_adapter"
            adapter_path.mkdir(parents=True, exist_ok=True)
            (adapter_path / "adapter_config.json").write_text('{}')

        checkpoint_io.save_checkpoint(checkpoint, path)
        assert (path / "trainer.pt").exists()
        loaded_checkpoint = checkpoint_io.load_checkpoint(path)
        assert 'state_dict' in loaded_checkpoint
    finally:
        for subdir in path.iterdir():
            if subdir.is_dir():
                for file in subdir.iterdir():
                    file.unlink()
                subdir.rmdir()
            else:
                subdir.unlink()
        path.rmdir()


def test_save_and_load_checkpoint(checkpoint_io, tmp_path):
    checkpoint = {'state_dict': {'layer.weight': torch.tensor([1.0])}}
    path = tmp_path / "checkpoint"
    save_and_load_checkpoint(checkpoint_io, checkpoint, path)


def test_save_and_load_checkpoint_adapter_only(adapter_checkpoint_io, tmp_path):
    checkpoint = {'state_dict': {'model.model.lora_a.weight': torch.tensor([1.0])}}
    path = tmp_path / "checkpoint"
    save_and_load_checkpoint(adapter_checkpoint_io, checkpoint, path, adapter_only=True)


def test_remove_checkpoint(checkpoint_io, tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "trainer.pt").touch()
    checkpoint_io.remove_checkpoint(path)
    assert not path.exists()
