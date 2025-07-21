# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest

from nemo.lightning.pytorch.strategies import MegatronStrategy


class TestMegatronStrategy:
    @patch('nemo.lightning.pytorch.strategies.megatron_strategy.create_checkpoint_io')
    def test_checkpoint_io(self, mock_create_checkpoint_io):
        class Dummy: ...

        mock_create_checkpoint_io.side_effect = lambda *args, **kwargs: Dummy()
        strategy = MegatronStrategy()

        first_io = strategy.checkpoint_io
        mock_create_checkpoint_io.assert_called_once()

        assert first_io == strategy.checkpoint_io

        new_io = object()
        strategy.checkpoint_io = new_io
        assert new_io == strategy.checkpoint_io

        strategy2 = MegatronStrategy()
        second_io = strategy2.checkpoint_io
        mock_create_checkpoint_io.assert_called()

        assert first_io != second_io
        assert second_io == strategy2.checkpoint_io

    def test_ckpt_load_main_params_and_ckpt_load_optimizer_both_true(self):
        # Make sure ckpt_load_optimizer and ckpt_load_main_params cannot be both set to True.
        with pytest.raises(ValueError):
            strategy = MegatronStrategy(ckpt_load_optimizer=True, ckpt_load_main_params=True)

    def test_ckpt_load_main_params_with_state_dict(self):
        # Test ckpt_load_main_params with "state_dict" key.
        strategy = MegatronStrategy()
        strategy.ckpt_load_main_params = True
        strategy.megatron_parallel = MagicMock()
        strategy.optimizers = [MagicMock()]
        checkpoint = {"state_dict": {"param": 1}}
        strategy.load_model_state_dict(checkpoint)
        strategy.optimizers[0].reload_model_params.assert_called_once_with(checkpoint["state_dict"])

    def test_ckpt_load_main_params_without_state_dict(self):
        # Test ckpt_load_main_params with "state_dict" key.
        strategy = MegatronStrategy()
        strategy.ckpt_load_main_params = True
        strategy.megatron_parallel = MagicMock()
        strategy.optimizers = [MagicMock()]
        checkpoint = {"other": 123}
        strategy.load_model_state_dict(checkpoint)
        strategy.optimizers[0].reload_model_params.assert_called_once_with(checkpoint)
