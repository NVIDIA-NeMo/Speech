import pytest
from lightning_fabric import plugins as fl_plugins
from lightning_fabric import strategies as fl_strategies
from pytorch_lightning import plugins as pl_plugins
from pytorch_lightning import strategies as pl_strategies

from nemo import lightning as nl
from nemo.lightning.fabric.conversion import to_fabric


class TestConversion:
    def test_ddp_strategy_conversion(self):
        pl_strategy = pl_strategies.DDPStrategy()
        fabric_strategy = to_fabric(pl_strategy)

        assert isinstance(fabric_strategy, fl_strategies.DDPStrategy)

    def test_fsdp_strategy_conversion(self):
        pl_strategy = pl_strategies.FSDPStrategy(
            cpu_offload=True,
        )
        fabric_strategy = to_fabric(pl_strategy)

        assert isinstance(fabric_strategy, fl_strategies.FSDPStrategy)
        assert fabric_strategy.cpu_offload.offload_params is True

    def test_mixed_precision_plugin_conversion(self):
        pl_plugin = pl_plugins.MixedPrecision(precision='16-mixed', device='cpu')
        fabric_plugin = to_fabric(pl_plugin)

        assert isinstance(fabric_plugin, fl_plugins.MixedPrecision)
        assert fabric_plugin.precision == '16-mixed'

    def test_fsdp_precision_plugin_conversion(self):
        pl_plugin = pl_plugins.FSDPPrecision(precision='16-mixed')
        fabric_plugin = to_fabric(pl_plugin)

        assert isinstance(fabric_plugin, fl_plugins.FSDPPrecision)
        assert fabric_plugin.precision == '16-mixed'

    def test_unsupported_object_conversion(self):
        class UnsupportedObject:
            pass

        with pytest.raises(NotImplementedError) as excinfo:
            to_fabric(UnsupportedObject())

        assert "No Fabric converter registered for UnsupportedObject" in str(excinfo.value)

    def test_megatron_strategy_conversion(self):
        pl_strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            context_parallel_size=2,
            sequence_parallel=True,
            expert_model_parallel_size=2,
            moe_extended_tp=True,
        )
        fabric_strategy = to_fabric(pl_strategy)

        assert isinstance(fabric_strategy, nl.FabricMegatronStrategy)
        assert fabric_strategy.tensor_model_parallel_size == 2
        assert fabric_strategy.pipeline_model_parallel_size == 2
        assert fabric_strategy.virtual_pipeline_model_parallel_size == 2
        assert fabric_strategy.context_parallel_size == 2
        assert fabric_strategy.sequence_parallel is True
        assert fabric_strategy.expert_model_parallel_size == 2
        assert fabric_strategy.moe_extended_tp is True

    def test_megatron_precision_conversion(self):
        pl_plugin = nl.MegatronMixedPrecision(precision='16-mixed')
        fabric_plugin = to_fabric(pl_plugin)

        assert isinstance(fabric_plugin, nl.FabricMegatronMixedPrecision)
        assert fabric_plugin.precision == '16-mixed'
