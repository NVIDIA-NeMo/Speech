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

from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Literal, Tuple, TypeVar, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.nn import Module
from torch.optim import Optimizer

from nemo.lightning._strategy_lib import GradScaler

AnyT = TypeVar("AnyT")


class MegatronMixedPrecision(MixedPrecision):
    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"],
        amp_O2: bool = False,
        device="cuda",
    ) -> None:
        if precision == "bf16-mixed":
            scaler = None
        else:
            scaler = GradScaler(init_scale=2**32, growth_interval=1000, hysteresis=2)

        super().__init__(precision, device, scaler)
        self.amp_O2 = amp_O2

    def connect(
        self, model: Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
    ) -> Tuple[Module, List[Optimizer], List[Any]]:
        """Connects this plugin to the accelerator and the training process."""
        from nemo.core.optim import MainParamsOptimizerWrapper

        if not optimizers or not self.amp_O2 or isinstance(optimizers[0], MainParamsOptimizerWrapper):
            return model, optimizers, lr_schedulers

        _optimizers = [*optimizers]
        _optimizers[0] = self.convert_optimizer(_optimizers[0])

        return model, _optimizers, lr_schedulers

    def convert_module(self, module: Module) -> Module:
        """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        from megatron.core.transformer.module import Float16Module
        from megatron.core.utils import get_model_config

        if self.precision in ["16-mixed", "bf16-mixed"]:
            config = get_model_config(module.module)
            config.fp16 = self.precision == "16-mixed"
            config.bf16 = self.precision == "bf16-mixed"
            if isinstance(module.module, Float16Module):
                new_float16_module = Float16Module(config, module.module.module)
                module.module = new_float16_module
            else:
                module.module = Float16Module(config, module.module)

        return module

    def convert_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Convert the optimizer parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        from nemo.core.optim import MainParamsOptimizerWrapper

        if isinstance(optimizer, MainParamsOptimizerWrapper) or not self.amp_O2:
            return optimizer

        return MainParamsOptimizerWrapper(
            optimizer,
            fp32_grad_accum=True,
            contiguous_grad_bucket=True,
        )

    def convert_input(self, data: AnyT) -> AnyT:
        """Convert model inputs (forward) to the floating point precision type of this plugin.

        Note: MegatronStrategy will take care of only doing this when:
            parallel_state.is_pipeline_first_stage()

        """
        return data

    def convert_output(self, data: AnyT) -> AnyT:
        """Convert outputs to the floating point precision type expected after model's forward.

        Note: MegatronStrategy will take care of only doing this when:
            parallel_state.is_pipeline_last_stage()

        """
        return data

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Union["pl.LightningModule", torch.nn.Module],
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        from nemo.core.optim import MainParamsOptimizerWrapper

        if not self.amp_O2 and not isinstance(optimizer, MainParamsOptimizerWrapper):
            return super().optimizer_step(optimizer, model, closure, **kwargs)

        if self.scaler is None:
            assert optimizer.fp32_grad_accumulation, "BF16 uses FP32 grad accumulation"
            _ = closure()
            self._after_closure(model, optimizer)
            return optimizer.step(**kwargs)

        assert not optimizer.fp32_grad_accumulation, "FP16 uses FP16 grad accumulation"
        closure_result = closure()

        # TODO: Add an option for merged all-reduce

        # cast fp16 grads to fp32 and copy to main grads, which are used for unscale and param update
        optimizer.copy_model_grads_to_main_grads()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        # unscale main (fp32) gradients
        self.scaler.unscale_(optimizer)
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            self.scaler.step(optimizer, **kwargs)
            self.scaler.update()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """No explicit precision casting. Inputs are supposed to be manually casted."""
        try:
            yield
        finally:
            pass


__all__ = ["MegatronMixedPrecision"]
