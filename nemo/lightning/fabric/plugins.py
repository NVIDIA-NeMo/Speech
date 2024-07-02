from contextlib import contextmanager
from typing import Any, Generator, Literal, Optional, TypeVar, Union

import torch
from lightning_fabric.plugins.precision import MixedPrecision
from lightning_fabric.utilities.types import Optimizable
from torch import nn
from torch.optim import Optimizer

from nemo.lightning._strategy_lib import GradScaler
from nemo.lightning.fabric.conversion import to_fabric
from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision

AnyT = TypeVar("AnyT")


class FabricMegatronMixedPrecision(MixedPrecision):
    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"] = "16-mixed",
        amp_02: bool = True,
        device="cuda",
        scaler: Optional[Union[torch.cuda.amp.GradScaler, str]] = None,
    ) -> None:
        if precision == "bf16-mixed":
            scaler = None
        else:
            scaler = GradScaler(
                init_scale=2**32,
                growth_interval=1000,
                hysteresis=2,
            )

        super().__init__(precision, device, scaler)
        self.amp_02 = amp_02

    def convert_input(self, data: AnyT) -> AnyT:
        """Convert model inputs (forward) to the floating point precision type of this plugin.

        Note: MegatronStrategy will take care of only doing this when:
            mpu.is_pipeline_first_stage()

        """
        return data

    def convert_output(self, data: AnyT) -> AnyT:
        """Convert outputs to the floating point precision type expected after model's forward.

        Note: MegatronStrategy will take care of only doing this when:
            mpu.is_pipeline_first_stage()

        """
        return data

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        from nemo.core.optim import MainParamsOptimizerWrapper

        return MainParamsOptimizerWrapper(
            optimizer,
            # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L496
            fp32_grad_accum=True,
            contiguous_grad_bucket=True,
        )

    def convert_module(self, module: nn.Module) -> nn.Module:
        """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        if not hasattr(module, "module"):
            return module

        from megatron.core.transformer.module import Float16Module
        from megatron.core.utils import get_model_config

        if self.precision in ["16-mixed", "bf16-mixed"]:
            config = get_model_config(module.module)
            config.fp16 = self.precision == "16-mixed"
            config.bf16 = self.precision == "bf16-mixed"
            if not isinstance(module.module, Float16Module):
                module.module = Float16Module(config, module.module)

        return module

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> None:
        from nemo.core.optim import MainParamsOptimizerWrapper

        assert isinstance(
            optimizer, MainParamsOptimizerWrapper
        ), "MegatronHalfPrecisionPlugin supports only the optimizer with master parameters"

        if self.scaler is None:
            assert optimizer.fp32_grad_accumulation, "BF16 uses FP32 grad accumulation"

            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(optimizer, **kwargs)

        assert not optimizer.fp32_grad_accumulation, "FP16 uses FP16 grad accumulation"

        # cast fp16 grads to fp32 and copy to main grads, which are used for unscale and param update
        optimizer.copy_model_grads_to_main_grads()

        # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
        step_output = self.scaler.step(optimizer, **kwargs)
        self.scaler.update()

        return step_output

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """No explicit precision casting. Inputs are supposed to be manually casted."""
        try:
            yield
        finally:
            pass


@to_fabric.register(MegatronMixedPrecision)
def _convert_megatron_mixed_precision(plugin: MegatronMixedPrecision) -> FabricMegatronMixedPrecision:
    return FabricMegatronMixedPrecision(
        precision=plugin.precision,
        device=plugin.device,
        scaler=plugin.scaler,
    )
