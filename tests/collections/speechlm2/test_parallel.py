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

import pytest
from lightning.pytorch.strategies.model_parallel import ModelParallelStrategy
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.moe.config import MoEParallelizerConfig
from omegaconf import DictConfig

from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy
from nemo.utils.trainer_utils import _resolve_automodel_configs, resolve_trainer_cfg


# ---------------------------------------------------------------------------
# AutomodelParallelStrategy
# ---------------------------------------------------------------------------


class TestAutomodelParallelStrategy:
    def test_is_subclass_of_model_parallel_strategy(self):
        assert issubclass(AutomodelParallelStrategy, ModelParallelStrategy)

    def test_default_init(self):
        strategy = AutomodelParallelStrategy()
        assert strategy._dp_size is None
        assert strategy._dp_replicate_size is None
        assert strategy._tp_size == 1
        assert strategy._pp_size == 1
        assert strategy._cp_size == 1
        assert strategy._ep_size == 1
        assert strategy._distributed_config is None
        assert strategy._moe_config is None
        assert strategy._moe_mesh is None
        assert strategy.activation_checkpointing_llm is False
        assert strategy.activation_checkpointing_perception is False

    def test_accepts_activation_checkpointing_flags(self):
        strategy = AutomodelParallelStrategy(
            activation_checkpointing_llm=True,
            activation_checkpointing_perception=True,
        )
        assert strategy.activation_checkpointing_llm is True
        assert strategy.activation_checkpointing_perception is True

    def test_activation_checkpointing_flags_are_independent(self):
        """Each AC flag can be set without the other."""
        llm_only = AutomodelParallelStrategy(activation_checkpointing_llm=True)
        assert llm_only.activation_checkpointing_llm is True
        assert llm_only.activation_checkpointing_perception is False

        perception_only = AutomodelParallelStrategy(activation_checkpointing_perception=True)
        assert perception_only.activation_checkpointing_llm is False
        assert perception_only.activation_checkpointing_perception is True

    def test_custom_parallelism_sizes(self):
        strategy = AutomodelParallelStrategy(
            dp_size=4,
            dp_replicate_size=2,
            tp_size=2,
            pp_size=2,
            cp_size=2,
            ep_size=4,
        )
        assert strategy._dp_size == 4
        assert strategy._dp_replicate_size == 2
        assert strategy._tp_size == 2
        assert strategy._pp_size == 2
        assert strategy._cp_size == 2
        assert strategy._ep_size == 4

    def test_accepts_distributed_config(self):
        cfg = FSDP2Config(sequence_parallel=True, defer_fsdp_grad_sync=False)
        strategy = AutomodelParallelStrategy(distributed_config=cfg)
        assert strategy.distributed_config is cfg
        assert strategy.distributed_config.sequence_parallel is True
        assert strategy.distributed_config.defer_fsdp_grad_sync is False

    def test_accepts_moe_config(self):
        cfg = MoEParallelizerConfig()
        strategy = AutomodelParallelStrategy(moe_config=cfg)
        assert strategy.moe_config is cfg

    def test_save_distributed_checkpoint_forwarded(self):
        strategy = AutomodelParallelStrategy(save_distributed_checkpoint=False)
        assert strategy._save_distributed_checkpoint is False

    def test_moe_mesh_initially_none(self):
        strategy = AutomodelParallelStrategy()
        assert strategy.moe_mesh is None

    def test_device_mesh_raises_before_setup(self):
        strategy = AutomodelParallelStrategy()
        with pytest.raises(RuntimeError):
            _ = strategy.device_mesh

    def test_distributed_sampler_kwargs_raises_before_setup(self):
        strategy = AutomodelParallelStrategy()
        with pytest.raises(RuntimeError):
            _ = strategy.distributed_sampler_kwargs


# ---------------------------------------------------------------------------
# _resolve_automodel_configs
# ---------------------------------------------------------------------------


class TestResolveAutomodelConfigs:
    """Tests for _resolve_automodel_configs which operates on an instantiated strategy object."""

    def test_plain_dict_to_fsdp2_config(self):
        strategy = AutomodelParallelStrategy(
            distributed_config={"defer_fsdp_grad_sync": False, "sequence_parallel": True},
        )
        _resolve_automodel_configs(strategy)
        assert isinstance(strategy.distributed_config, FSDP2Config)
        assert strategy.distributed_config.defer_fsdp_grad_sync is False
        assert strategy.distributed_config.sequence_parallel is True
        # Check that __post_init__ created the default mp_policy
        assert strategy.distributed_config.mp_policy is not None

    def test_plain_dict_to_moe_config(self):
        strategy = AutomodelParallelStrategy(
            moe_config={"reshard_after_forward": True},
        )
        _resolve_automodel_configs(strategy)
        assert isinstance(strategy.moe_config, MoEParallelizerConfig)
        assert strategy.moe_config.reshard_after_forward is True

    def test_noop_when_no_configs(self):
        """Strategy without configs (e.g. DDPStrategy) is untouched."""
        from lightning.pytorch.strategies import DDPStrategy

        strategy = DDPStrategy()
        _resolve_automodel_configs(strategy)  # should not raise

    def test_noop_when_already_objects(self):
        original_cfg = FSDP2Config()
        original_moe = MoEParallelizerConfig()
        strategy = AutomodelParallelStrategy(
            distributed_config=original_cfg,
            moe_config=original_moe,
        )
        _resolve_automodel_configs(strategy)
        assert strategy.distributed_config is original_cfg
        assert strategy.moe_config is original_moe

    def test_noop_when_configs_are_none(self):
        strategy = AutomodelParallelStrategy()
        _resolve_automodel_configs(strategy)
        assert strategy.distributed_config is None
        assert strategy.moe_config is None

    def test_empty_dict_creates_default_configs(self):
        strategy = AutomodelParallelStrategy(distributed_config={}, moe_config={})
        _resolve_automodel_configs(strategy)
        assert isinstance(strategy.distributed_config, FSDP2Config)
        assert isinstance(strategy.moe_config, MoEParallelizerConfig)
        # All defaults should apply
        assert strategy.distributed_config.defer_fsdp_grad_sync is True
        assert strategy.distributed_config.sequence_parallel is False

    def test_nested_target_in_distributed_config(self):
        """A sub-field like mp_policy can use _target_ for Hydra instantiation."""
        strategy = AutomodelParallelStrategy(
            distributed_config={
                "sequence_parallel": True,
                "mp_policy": {
                    "_target_": "torch.distributed.fsdp.MixedPrecisionPolicy",
                    "param_dtype": "torch.bfloat16",
                },
            },
        )
        _resolve_automodel_configs(strategy)
        assert isinstance(strategy.distributed_config, FSDP2Config)
        assert strategy.distributed_config.sequence_parallel is True
        from torch.distributed.fsdp import MixedPrecisionPolicy

        assert isinstance(strategy.distributed_config.mp_policy, MixedPrecisionPolicy)

    def test_both_configs_resolved_together(self):
        strategy = AutomodelParallelStrategy(
            distributed_config={"sequence_parallel": False},
            moe_config={},
        )
        _resolve_automodel_configs(strategy)
        assert isinstance(strategy.distributed_config, FSDP2Config)
        assert isinstance(strategy.moe_config, MoEParallelizerConfig)


# ---------------------------------------------------------------------------
# resolve_trainer_cfg (end-to-end with AutomodelParallelStrategy)
# ---------------------------------------------------------------------------


class TestResolveTrainerCfg:
    def test_automodel_strategy_from_yaml(self):
        """End-to-end: YAML dict config -> instantiated AutomodelParallelStrategy."""
        trainer_cfg = DictConfig(
            {
                "devices": 1,
                "accelerator": "cpu",
                "strategy": {
                    "_target_": "nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy",
                    "tp_size": 2,
                    "pp_size": 1,
                    "distributed_config": {
                        "sequence_parallel": True,
                        "defer_fsdp_grad_sync": False,
                    },
                    "moe_config": {},
                },
            }
        )
        resolved = resolve_trainer_cfg(trainer_cfg)
        strategy = resolved["strategy"]
        assert isinstance(strategy, AutomodelParallelStrategy)
        assert strategy._tp_size == 2
        assert strategy._pp_size == 1
        assert isinstance(strategy.distributed_config, FSDP2Config)
        assert strategy.distributed_config.sequence_parallel is True
        assert strategy.distributed_config.defer_fsdp_grad_sync is False
        assert isinstance(strategy.moe_config, MoEParallelizerConfig)

    def test_non_automodel_strategy_unaffected(self):
        """Other strategies (e.g. DDPStrategy) should pass through unchanged."""
        trainer_cfg = DictConfig(
            {
                "devices": 1,
                "accelerator": "cpu",
                "strategy": {
                    "_target_": "lightning.pytorch.strategies.DDPStrategy",
                    "gradient_as_bucket_view": True,
                    "find_unused_parameters": True,
                },
            }
        )
        resolved = resolve_trainer_cfg(trainer_cfg)
        from lightning.pytorch.strategies import DDPStrategy

        assert isinstance(resolved["strategy"], DDPStrategy)

    def test_string_strategy_unaffected(self):
        """A plain string strategy (e.g. 'ddp') should pass through."""
        trainer_cfg = DictConfig(
            {
                "devices": 1,
                "accelerator": "cpu",
                "strategy": "ddp",
            }
        )
        resolved = resolve_trainer_cfg(trainer_cfg)
        assert resolved["strategy"] == "ddp"

    def test_automodel_strategy_with_ac_flags_from_yaml(self):
        """Both AC flags can be set via YAML and survive Hydra instantiation."""
        trainer_cfg = DictConfig(
            {
                "devices": 1,
                "accelerator": "cpu",
                "strategy": {
                    "_target_": "nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy",
                    "ep_size": 4,
                    "activation_checkpointing_llm": True,
                    "activation_checkpointing_perception": True,
                },
            }
        )
        resolved = resolve_trainer_cfg(trainer_cfg)
        strategy = resolved["strategy"]
        assert isinstance(strategy, AutomodelParallelStrategy)
        assert strategy.activation_checkpointing_llm is True
        assert strategy.activation_checkpointing_perception is True

    def test_automodel_strategy_without_configs(self):
        """AutomodelParallelStrategy can be specified without distributed_config/moe_config."""
        trainer_cfg = DictConfig(
            {
                "devices": 1,
                "accelerator": "cpu",
                "strategy": {
                    "_target_": "nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy",
                    "tp_size": 4,
                    "ep_size": 8,
                },
            }
        )
        resolved = resolve_trainer_cfg(trainer_cfg)
        strategy = resolved["strategy"]
        assert isinstance(strategy, AutomodelParallelStrategy)
        assert strategy._tp_size == 4
        assert strategy._ep_size == 8
        # No configs passed → still None (will be defaulted in setup_environment)
        assert strategy._distributed_config is None
        assert strategy._moe_config is None
