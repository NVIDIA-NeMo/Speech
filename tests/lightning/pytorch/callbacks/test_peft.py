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

from unittest.mock import MagicMock, call, patch

import torch
import torch.nn as nn
from lightning.pytorch.trainer.states import TrainerFn

from nemo.collections.llm import fn
from nemo.lightning.pytorch.callbacks.peft import PEFT, WrappedAdapterIO
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO


class TestPEFT:
    class DummyPEFT(PEFT):
        def transform(self, module, name=None, prefix=None):
            return module  # No-op transform for testing

        def freeze_model(self, module):
            super().freeze_model(module)
            self.is_called = True
            return module

    class DummyModel(nn.Module, fn.FNMixin):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.conv = nn.Conv2d(3, 3, 3)

    def test_peft_call(self):
        model = self.DummyModel()
        peft = self.DummyPEFT()

        transformed_model = peft(model)

        assert (
            hasattr(peft, "is_called") and peft.is_called == True
        ), "peft methods may subclass `freeze_model()`, so it must be called"
        assert transformed_model.linear.weight.requires_grad == False
        assert transformed_model.conv.weight.requires_grad == False

    def test_linear_adapter(self):
        from nemo.collections.llm.peft.lora import LinearAdapter

        for has_bias in [True, False]:
            linear = nn.Linear(10, 10, bias=has_bias)
            linear_adapter = LinearAdapter(linear)
            bias_in_state_dict = 'bias' in linear.state_dict()
            if has_bias:
                assert bias_in_state_dict
            else:
                assert not bias_in_state_dict

            # Check if the state-dict keys changed
            for key, val in linear.state_dict().items():
                assert key in linear_adapter.state_dict(), f"Key {key} not found in LinearAdapter"
                assert torch.equal(val, linear_adapter.state_dict()[key]), f"Key {key} diff. val in LinearAdapter"
            # Make sure the additional keys are in the allow list
            for key, val in linear_adapter.state_dict().items():
                if key in linear.state_dict():
                    continue
                assert key in ['lora_a', 'lora_b']

    def test_linear_adapter_monkey_patch(self):
        from copy import deepcopy

        from nemo.collections.llm.peft.lora import patch_linear_module

        linear = nn.Linear(10, 10)
        state_init = deepcopy(linear.state_dict())
        linear_adapter = patch_linear_module(linear)
        # Check if the state-dict keys changed
        for key, val in state_init.items():
            assert key in linear_adapter.state_dict(), f"Key {key} not found in LinearAdapter"
            assert torch.equal(val, linear_adapter.state_dict()[key]), f"Key {key} diff. val in LinearAdapter"
        # Make sure the additional keys are in the allow list
        for key, val in linear_adapter.state_dict().items():
            if key in state_init:
                continue
            assert key in ['lora_a', 'lora_b']

        for key in ['lora_a', 'lora_b']:
            assert hasattr(linear_adapter, key), f"Expected {key} to be in module"
            assert key in linear_adapter.state_dict(), f"Expected {key} to be in state dict"
            assert getattr(linear_adapter, key).requires_grad == True, "Expected {key} to require_grad"

    def test_peft_setup(self):
        peft = self.DummyPEFT()
        trainer = MagicMock()
        pl_module = MagicMock()

        pl_module.model_transform = peft
        peft.setup(trainer, pl_module, "fit")

        assert isinstance(trainer.strategy._checkpoint_io, AsyncFinalizableCheckpointIO)
        assert isinstance(trainer.strategy._checkpoint_io._checkpoint_io, WrappedAdapterIO)
        assert peft.model_transform is not None
        assert peft._needs_to_call is True

    @patch('nemo.lightning.pytorch.callbacks.peft.logging')
    def test_peft_on_train_epoch_start_with_adapter(self, mock_logging):
        peft = self.DummyPEFT()
        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.model_transform = peft
        trainer.state.fn = TrainerFn.FITTING  # Mock the trainer to be in FITTING state

        peft.setup(trainer, pl_module, "fit")

        assert peft.model_transform is not None
        assert peft._needs_to_call is True

        peft.wrapped_io = MagicMock()
        peft.wrapped_io.adapter_ckpt_path = "dummy_path"
        peft.wrapped_io.load_checkpoint.return_value = {"dummy_state": "dummy_value"}
        peft.on_train_epoch_start(trainer, pl_module)

        # Check for all expected log messages
        mock_logging.info.assert_has_calls(
            [
                call("Loading adapters from dummy_path"),
                call("Initializing model parallel"),
                call("Setting up optimizers"),
            ],
            any_order=True,
        )

        # Verify the number of calls
        assert mock_logging.info.call_count == 3

        trainer.strategy.load_model_state_dict.assert_called_once_with({"dummy_state": "dummy_value"}, strict=False)
        trainer.strategy.init_model_parallel.assert_called_once()
        trainer.strategy.setup_optimizers.assert_called_once_with(trainer)
