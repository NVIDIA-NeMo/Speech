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

from megatron.core.distributed import DistributedDataParallelConfig

from nemo.lightning.pytorch.strategies import MegatronStrategy


def get_metadata(
    ckpt_parallel_save_optim: bool = None,
    ckpt_optim_fully_reshardable: bool = None,
) -> dict:
    metadata = {
        'singleton_local_shards': False,
        'chained_optim_avoid_prefix': True,
    }
    if ckpt_optim_fully_reshardable:
        metadata['distrib_optim_sharding_type'] = 'fully_reshardable'
        metadata['distrib_optim_fully_reshardable_mem_efficient'] = False
    else:
        metadata['distrib_optim_sharding_type'] = 'dp_reshardable'

    return metadata


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

    def test_sharded_state_dict_metadata(self):
        strategy = MegatronStrategy(ckpt_parallel_save_optim=True)

        ddp = DistributedDataParallelConfig(use_distributed_optimizer=True)

        strategy = MegatronStrategy(ckpt_optim_fully_reshardable=True, ddp=ddp)
        metadata = strategy.sharded_state_dict_metadata
        assert metadata == get_metadata(ckpt_optim_fully_reshardable=True)

        strategy = MegatronStrategy(ddp=ddp)
        metadata = strategy.sharded_state_dict_metadata
        assert metadata == get_metadata()

    def test_init_errors(self):
        with pytest.raises(ValueError):
            strategy = MegatronStrategy(ddp="pytorch", fsdp="fsdp")

        with pytest.raises(ValueError):
            strategy = MegatronStrategy(ddp="test")

        with pytest.raises(NotImplementedError):
            strategy = MegatronStrategy(fsdp="pytorch")

        ddp = DistributedDataParallelConfig(use_custom_fsdp=True)
        strategy = MegatronStrategy(ddp=ddp, fsdp=None)

        ddp = DistributedDataParallelConfig(use_custom_fsdp=False)
        strategy = MegatronStrategy(ddp=ddp, fsdp="megatron")

        with pytest.raises(ValueError):
            strategy = MegatronStrategy(fsdp="test")

    def test_process_dataloader(self):
        strategy = MegatronStrategy()
        strategy.process_dataloader(dataloader="test")

    def test_on_test_end(self):
        strategy = MegatronStrategy()
        strategy.model = [1, 2, 3]
        with pytest.raises(AttributeError):
            strategy.on_test_end()

    def test_update_step_kwargs(self):
        strategy = MegatronStrategy()
        with pytest.raises(AttributeError):
            strategy._update_step_kwargs(1, kwargs={"forward_step": None}, step_name="first")

        with pytest.raises(AttributeError):
            strategy._update_step_kwargs(1, kwargs={"data_step": None}, step_name="first")

        with pytest.raises(AttributeError):
            strategy._update_step_kwargs(1, kwargs={"data_step": None, "forward_step": None}, step_name="first")

    def test_create_all_gather_group_default(self):
        """Test that create_all_gather_group defaults to False."""
        strategy = MegatronStrategy()
        assert strategy.create_all_gather_group == False

    def test_create_all_gather_group_enabled(self):
        """Test that create_all_gather_group can be set to True."""
        strategy = MegatronStrategy(create_all_gather_group=True)
        assert strategy.create_all_gather_group == True

    def test_create_all_gather_group_in_parallelism_config(self):
        """Test that create_all_gather_group can be configured via ParallelismConfig."""
        import torch

        from nemo.lightning.pytorch.strategies.megatron_strategy import ParallelismConfig

        parallel_config = ParallelismConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=None,
            microbatch_group_size_per_vp_stage=1,
            context_parallel_size=1,
            sequence_parallel=False,
            expert_model_parallel_size=1,
            moe_extended_tp=False,
            pipeline_dtype=torch.float32,
            create_all_gather_group=True,
        )

        assert parallel_config.create_all_gather_group == True

        # Test default value
        parallel_config_default = ParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            microbatch_group_size_per_vp_stage=1,
            context_parallel_size=1,
            sequence_parallel=False,
            expert_model_parallel_size=1,
            moe_extended_tp=False,
            pipeline_dtype=torch.float32,
        )
        assert parallel_config_default.create_all_gather_group == False
