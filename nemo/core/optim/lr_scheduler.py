# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import dataclasses
import inspect
import math
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union

import hydra
import torch.optim as optim
import torch.optim.lr_scheduler as pt_scheduler
import torch.utils.data.dataloader as dataloader
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler

from nemo.core.config import SchedulerParams, get_scheduler_config, register_scheduler_params
from nemo.utils import logging
from nemo.utils.model_utils import maybe_update_config_version


class WarmupPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(self, optimizer, *, warmup_steps=None, warmup_ratio=None, max_steps=None, min_lr=0.0, last_epoch=-1):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        if (self.max_steps is not None) and (step > self.max_steps):
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class SquareRootConstantPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        constant_steps=None,
        constant_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (
            constant_steps is not None and constant_ratio is not None
        ), "Either use particular number of step or ratio"
        assert constant_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if constant_steps is not None:
            self.constant_steps = constant_steps
        elif constant_ratio is not None:
            self.constant_steps = int(constant_ratio * max_steps)
        else:
            self.constant_steps = 0

        self.constant_lr = 1 / (constant_steps**0.5)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        if step <= self.constant_steps:
            return [self.constant_lr for _ in self.base_lrs]

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class WarmupHoldPolicy(WarmupPolicy):
    """Variant of WarmupPolicy which maintains high learning rate for a defined number of steps.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        hold_steps: Number of training steps to hold the learning rate after warm up
        hold_ratio: Ratio of hold steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        hold_steps=None,
        hold_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (hold_steps is not None and hold_ratio is not None), "Either use particular number of step or ratio"
        assert hold_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        self.min_lr = min_lr
        self._last_warmup_lr = 0.0

        # Necessary to duplicate as class attributes are hidden in inner class
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * max_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

        super().__init__(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        # Warmup phase
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        # Hold phase
        if (step >= self.warmup_steps) and (step < self.hold_steps):
            return self.base_lrs

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)


class WarmupAnnealHoldPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
        min_lr: Minimum lr to hold the learning rate after decay at.
        constant_steps: Number of steps to keep lr constant at.
        constant_ratio: Ratio of steps to keep lr constant.
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        constant_steps=None,
        constant_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert not (
            constant_steps is not None and constant_ratio is not None
        ), "Either use constant_steps or constant_ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps

        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if constant_steps is not None:
            self.constant_steps = constant_steps
        elif constant_ratio is not None:
            self.constant_steps = int(constant_ratio * max_steps)
        else:
            self.constant_steps = 0

        self.decay_steps = max_steps - (self.constant_steps + self.warmup_steps)

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        # Reset learning rate
        if 'reset_lr' in self.optimizer.param_groups[0].keys():
            reset_lr = self.optimizer.param_groups[0]['reset_lr']
            num_steps = reset_lr['num_steps']
            step -= num_steps
            if reset_lr['if_init_step'] and reset_lr['reset_lr_steps']:
                self.decay_steps -= num_steps
                self.max_steps -= num_steps
                self.optimizer.param_groups[0]['reset_lr']['if_init_step'] = False

        # Warmup steps
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self._get_warmup_lr(step)

        # Constant steps after warmup and decay
        if self.constant_steps > 0 and (self.warmup_steps + self.decay_steps) < step <= self.max_steps:
            return self._get_constant_lr(step)

        # Min lr after max steps of updates
        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_constant_lr(self, step):
        return [self.min_lr for _ in self.base_lrs]

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


def _squareroot_annealing(initial_lr, step, max_steps, min_lr):
    mult = ((max_steps - step) / max_steps) ** 0.5
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _square_annealing(initial_lr, step, max_steps, min_lr):
    mult = ((max_steps - step) / max_steps) ** 2
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _cosine_annealing(initial_lr, step, max_steps, min_lr):
    mult = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    out_lr = (initial_lr - min_lr) * mult + min_lr
    return out_lr


def _linear_warmup_with_cosine_annealing(max_lr, warmup_steps, step, decay_steps, min_lr):

    assert max_lr > min_lr
    # Use linear warmup for the initial part.
    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * float(step) / float(warmup_steps)

    # For any steps larger than `decay_steps`, use `min_lr`.
    if step > warmup_steps + decay_steps:
        return min_lr

    # If we are done with the warmup period, use the decay style.
    num_steps_ = step - warmup_steps
    decay_steps_ = decay_steps
    decay_ratio = float(num_steps_) / float(decay_steps_)
    assert decay_ratio >= 0.0
    assert decay_ratio <= 1.0
    delta_lr = max_lr - min_lr

    coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)

    return min_lr + coeff * delta_lr


def _poly_decay(initial_lr, step, decay_steps, power, min_lr, cycle):
    if cycle:
        multiplier = 1.0 if step == 0 else math.ceil(step / decay_steps)
        decay_steps *= multiplier
    else:
        step = min(step, decay_steps)
    p = step / decay_steps
    lr = (initial_lr - min_lr) * math.pow(1.0 - p, power)
    lr += min_lr
    return lr


def _noam_hold_annealing(initial_lr, step, warmup_steps, hold_steps, decay_rate, min_lr):
    # hold_steps = total number of steps to hold the LR, not the warmup + hold steps.
    T_warmup_decay = max(1, warmup_steps**decay_rate)
    T_hold_decay = max(1, (step - hold_steps) ** decay_rate)
    lr = (initial_lr * T_warmup_decay) / T_hold_decay
    lr = max(lr, min_lr)
    return lr


class SquareAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=1e-5, last_epoch=-1, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _square_annealing(
                initial_lr=initial_lr,
                step=step - self.warmup_steps,
                max_steps=self.max_steps - self.warmup_steps,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class SquareRootAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0, last_epoch=-1, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _squareroot_annealing(initial_lr=initial_lr, step=step, max_steps=self.max_steps, min_lr=self.min_lr)
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class CosineAnnealing(WarmupAnnealHoldPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0, last_epoch=-1, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the minimum learning rate."
                )

        if self.constant_steps is None or self.constant_steps == 0:
            new_lrs = [
                _cosine_annealing(
                    initial_lr=initial_lr,
                    step=step - self.warmup_steps,
                    max_steps=self.max_steps - self.warmup_steps,
                    min_lr=self.min_lr,
                )
                for initial_lr in self.base_lrs
            ]
        else:
            new_lrs = self._get_linear_warmup_with_cosine_annealing_lr(step)
        return new_lrs

    def _get_warmup_lr(self, step):
        if self.constant_steps is None or self.constant_steps == 0:
            return super()._get_warmup_lr(step)
        else:
            # Use linear warmup for the initial part.
            return self._get_linear_warmup_with_cosine_annealing_lr(step)

    def _get_constant_lr(self, step):
        # Only called when `constant_steps` > 0.
        return self._get_linear_warmup_with_cosine_annealing_lr(step)

    def _get_linear_warmup_with_cosine_annealing_lr(self, step):
        # Cosine Schedule for Megatron LM, slightly different warmup schedule + constant LR at the end.
        new_lrs = [
            _linear_warmup_with_cosine_annealing(
                max_lr=self.base_lrs[0],
                warmup_steps=self.warmup_steps,
                step=step,
                decay_steps=self.decay_steps,
                min_lr=self.min_lr,
            )
            for _ in self.base_lrs
        ]
        return new_lrs


class NoamAnnealing(_LRScheduler):
    def __init__(
        self,
        optimizer,
        *,
        d_model,
        warmup_steps=None,
        warmup_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        self._normalize = d_model ** (-0.5)
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the minimum learning rate."
                )

        new_lrs = [self._noam_annealing(initial_lr=initial_lr, step=step) for initial_lr in self.base_lrs]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        if self.warmup_steps > 0:
            mult = self._normalize * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        else:
            mult = self._normalize * step ** (-0.5)

        out_lr = initial_lr * mult
        if step > self.warmup_steps:
            out_lr = max(out_lr, self.min_lr)
        return out_lr


class NoamHoldAnnealing(WarmupHoldPolicy):
    def __init__(self, optimizer, *, max_steps, decay_rate=0.5, min_lr=0.0, last_epoch=-1, **kwargs):
        """
        Implementation of the Noam Hold Annealing policy from the SqueezeFormer paper.

        Unlike NoamAnnealing, the peak learning rate can be explicitly set for this scheduler.
        The schedule first performs linear warmup, then holds the peak LR, then decays with some schedule for
        the remainder of the steps. Therefore the min-lr is still dependent on the hyper parameters selected.

        It's schedule is determined by three factors-

        Warmup Steps: Initial stage, where linear warmup occurs uptil the peak LR is reached. Unlike NoamAnnealing,
            the peak LR is explicitly stated here instead of a scaling factor.

        Hold Steps: Intermediate stage, where the peak LR is maintained for some number of steps. In this region,
            the high peak LR allows the model to converge faster if training is stable. However the high LR
            may also cause instability during training. Should usually be a significant fraction of training
            steps (around 30-40% of the entire training steps).

        Decay Steps: Final stage, where the LR rapidly decays with some scaling rate (set by decay rate).
            To attain Noam decay, use 0.5, for Squeezeformer recommended decay, use 1.0. The fast decay after
            prolonged high LR during hold phase allows for rapid convergence.

        References:
            - [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](https://arxiv.org/abs/2206.00888)

        Args:
            optimizer: Pytorch compatible Optimizer object.
            warmup_steps: Number of training steps in warmup stage
            warmup_ratio: Ratio of warmup steps to total steps
            hold_steps: Number of training steps to hold the learning rate after warm up
            hold_ratio: Ratio of hold steps to total steps
            max_steps: Total number of steps while training or `None` for
                infinite training
            decay_rate: Float value describing the polynomial decay after the hold period. Default value
                of 0.5 corresponds to Noam decay.
            min_lr: Minimum learning rate.
        """
        self.decay_rate = decay_rate
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        if self.warmup_steps is None or self.warmup_steps == 0:
            raise ValueError("Noam scheduler cannot be used without warmup steps")

        if self.hold_steps > 0:
            hold_steps = self.hold_steps - self.warmup_steps
        else:
            hold_steps = 0

        new_lrs = [
            _noam_hold_annealing(
                initial_lr,
                step=step,
                warmup_steps=self.warmup_steps,
                hold_steps=hold_steps,
                decay_rate=self.decay_rate,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class WarmupAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, last_epoch=-1, min_lr=0.0, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        delta_lr = self.base_lrs[0] - self.min_lr
        mult = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        out_lr = [self.min_lr + (1 - mult) * delta_lr for _ in self.base_lrs]
        return out_lr


class InverseSquareRootAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, last_epoch=-1, min_lr=0.0, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, **kwargs, last_epoch=last_epoch, min_lr=min_lr)

    def _get_lr(self, step):
        denom = ((step + 1) / (self.warmup_steps + 1)) ** 0.5
        out_lr = [initial_lr / denom for initial_lr in self.base_lrs]
        return out_lr


class T5InverseSquareRootAnnealing(SquareRootConstantPolicy):
    def __init__(self, optimizer, *, max_steps, last_epoch=-1, min_lr=0.0, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, **kwargs, last_epoch=last_epoch, min_lr=min_lr)

    def _get_lr(self, step):
        return [1 / (step**0.5) for _ in self.base_lrs]


class PolynomialDecayAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0.0, power=1.0, cycle=False, last_epoch=-1, **kwargs):
        self.power = power
        self.cycle = cycle

        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _poly_decay(
                initial_lr,
                step=step - self.warmup_steps,
                decay_steps=self.max_steps - self.warmup_steps,
                power=self.power,
                min_lr=self.min_lr,
                cycle=self.cycle,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class PolynomialHoldDecayAnnealing(WarmupHoldPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0.0, power=1.0, cycle=False, last_epoch=-1, **kwargs):
        self.power = power
        self.cycle = cycle

        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _poly_decay(
                initial_lr,
                step=step - self.hold_steps,
                decay_steps=self.max_steps - max(self.warmup_steps, self.hold_steps),
                power=self.power,
                min_lr=self.min_lr,
                cycle=self.cycle,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


def register_scheduler(name: str, scheduler: _LRScheduler, scheduler_params: SchedulerParams):
    """
    Checks if the scheduler name exists in the registry, and if it doesnt, adds it.

    This allows custom schedulers to be added and called by name during instantiation.

    Args:
        name: Name of the optimizer. Will be used as key to retrieve the optimizer.
        scheduler: Scheduler class (inherits from _LRScheduler)
        scheduler_params: The parameters as a dataclass of the scheduler
    """
    if name in AVAILABLE_SCHEDULERS:
        raise ValueError(f"Cannot override pre-existing schedulers. Conflicting scheduler name = {name}")

    AVAILABLE_SCHEDULERS[name] = scheduler

    sched_name = "{}_params".format(scheduler.__name__)
    register_scheduler_params(name=sched_name, scheduler_params=scheduler_params)


def get_scheduler(name: str, **kwargs: Optional[Dict[str, Any]]) -> _LRScheduler:
    """
    Convenience method to obtain an _LRScheduler class and partially instantiate it with optimizer kwargs.

    Args:
        name: Name of the scheduler in the registry.
        kwargs: Optional kwargs of the scheduler used during instantiation.

    Returns:
        a partially instantiated _LRScheduler
    """
    if name not in AVAILABLE_SCHEDULERS:
        raise ValueError(
            f"Cannot resolve scheduler{name}'. Available optimizers are : " f"{AVAILABLE_SCHEDULERS.keys()}"
        )

    scheduler_cls = AVAILABLE_SCHEDULERS[name]
    # Pop 'max_steps' if it's not required by the scheduler
    if 'max_steps' in kwargs and 'max_steps' not in inspect.signature(scheduler_cls).parameters:
        kwargs.pop('max_steps')
    scheduler = partial(scheduler_cls, **kwargs)
    return scheduler


def prepare_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: Union[Dict[str, Any], DictConfig],
    train_dataloader: Optional[dataloader.DataLoader] = None,
) -> Optional[Dict[str, Any]]:
    """
    Constructs an LR Scheduler (optionally) for a given optimizer, based on a config with the following schema

    optim:
      name: <name of optimizer>
      lr: <maximal learning rate>

      # <additional optimizer arguments>
      args:
        name: auto  # special keyword, resolves to correct optimizer config for given optimizer name
        # cls: nemo.core.config.optimizers.NovogradParams  # explicit instantiation by class path
        params:  # optional override parameters for the optimizer config
          betas: [0.8, 0.5]
          weight_decay: 0.001

      # scheduler setup
      sched:
        name: <name of scheduler>
        iters_per_batch: null # computed at runtime; mandatory to have
        max_steps: -1 # computed at runtime or explicitly set here; mandatory to have

        # pytorch lightning args <mandatory>
        monitor: val_loss
        reduce_on_plateau: false

        # <scheduler config override>
        args:
          name: auto  # special keyword, resolves to correct optimizer config for given optimizer name
          # cls: nemo.core.config.schedulers.CosineAnnealingParams  # explicit instantiation by class path
          params:  # optional override parameters for the optimizer config
            warmup_steps: null
            warmup_ratio: null
            min_lr: 0.0
            last_epoch: -1

    Args:
        optimizer: An instantiated Optimizer.
        scheduler_config: A dictionary / config dict which follows the above schema.
        train_dataloader: Optional requirement, must be passed if "iters_per_batch" is defined
            instead of "max_steps". Used to compute effective "max_steps".

    Returns:
        A dictionary containing the LR Scheduler implementation if the config was successfully parsed
        along with other parameters required by Pytorch Lightning, otherwise None.
    """
    if scheduler_config is not None:
        scheduler_config = maybe_update_config_version(scheduler_config)

    # Build nested dictionary for convenience out of structured objects
    if isinstance(scheduler_config, DictConfig):
        scheduler_config = OmegaConf.to_container(scheduler_config, resolve=True)

    elif dataclasses.is_dataclass(scheduler_config):
        # Recursively transform data classes to basic dictionaries
        scheduler_config = OmegaConf.create(scheduler_config)
        scheduler_config = OmegaConf.to_container(scheduler_config, resolve=True)

    # Test to see if config follows above schema
    interval = 'step'
    if scheduler_config is not None:
        if 'args' in scheduler_config:
            scheduler_args = scheduler_config.pop('args')
        else:
            scheduler_args = copy.deepcopy(scheduler_config)

            # Remove extra parameters from scheduler_args nest
            # Assume all other parameters are to be passed into scheduler constructor
            scheduler_args.pop('name', None)
            scheduler_args.pop('t_max_epochs', None)
            scheduler_args.pop('t_accumulate_grad_batches', None)
            scheduler_args.pop('t_limit_train_batches', None)
            scheduler_args.pop('t_num_workers', None)
            scheduler_args.pop('monitor', None)
            scheduler_args.pop('reduce_on_plateau', None)

        if 'name' in scheduler_config and scheduler_config['name'] in EPOCH_SCHEDULERS:
            interval = 'epoch'

    else:
        # Return gracefully in case `sched` was not supplied; inform user
        logging.info('Scheduler not initialized as no `sched` config supplied to setup_optimizer()')
        return None

    # Try instantiation of scheduler params from config class path
    if '_target_' in scheduler_args:
        scheduler_args_cfg = OmegaConf.create(scheduler_args)
        scheduler_conf = hydra.utils.instantiate(scheduler_args_cfg)
        scheduler_args = vars(scheduler_conf)

        # Get name of the scheduler
        scheduler_name = scheduler_conf.__class__.__name__

        if 'Params' in scheduler_name:
            scheduler_name = scheduler_name.replace('Params', '')

    else:
        # Class path instantiation failed; try resolving "name" component

        # Get name of the scheduler
        if 'name' in scheduler_config:
            scheduler_name = scheduler_config['name']
        else:
            logging.warning(
                "Could not resolve classpath for Scheduler Config, and `name` "
                "was not provided either. \n"
                "Scheduler cannot be instantiated !"
            )
            return None

        # If class path was not provided, perhaps `name` is provided for resolution
        if 'name' in scheduler_args:
            # If `auto` is passed as name for resolution of optimizer name,
            # then lookup optimizer name and resolve its parameter config
            if scheduler_args['name'] == 'auto':
                scheduler_params_name = "{}Params".format(scheduler_name)
            else:
                scheduler_params_name = scheduler_args['name']

            # Get override arguments provided in the config yaml file / Dict Config
            scheduler_params_override = scheduler_args.get('params', {})

            # If params is itself a dict config object provided explicitly in Dict Config
            # Resolve to dictionary for convenience
            if isinstance(scheduler_params_override, DictConfig):
                scheduler_params_override = OmegaConf.to_container(scheduler_params_override, resolve=True)

            # Get and instantiate the Config dataclass for this scheduler
            scheduler_params_cls = get_scheduler_config(scheduler_params_name, **scheduler_params_override)
            scheduler_params = scheduler_params_cls()  # instantiate the parameters object
            scheduler_args = vars(scheduler_params)  # extract just the dictionary from the Config object

        else:
            # assume the input dictionary is schedular args (from dataclasses / omegaconf)
            pass

    # Extract value to monitor in losses, if provided.
    if 'monitor' in scheduler_config:
        monitor = scheduler_config.get('monitor')
    else:
        # Default to train loss
        monitor = 'loss'

    # Store exact max_steps if it is provided
    max_steps_from_cfg = scheduler_config.get('max_steps')
    if max_steps_from_cfg is not None:
        if max_steps_from_cfg == -1:
            logging.warning('`max_steps` is set to -1 in the scheduler config, scheduler will not be instantiated')
            return None
        assert max_steps_from_cfg >= 0, "`max_steps` must be a non-negative integer"
        max_steps = max_steps_from_cfg

    elif 't_max_epochs' in scheduler_config:
        # Compute effective max_steps if t_max_epochs is provided
        if train_dataloader is None:
            logging.warning(
                'As `t_max_epochs` is provided/computed, it is required to pass the train dataloader in order\n'
                'to compute effective maximum number of steps.\n'
                'Scheduler will not be instantiated !'
            )
            return None

        # Raise exception if neither `max_steps` nor `t_max_epochs` is provided
        if scheduler_config.get('t_max_epochs', None) is None:
            logging.warning(
                "`t_max_epochs` cannot be None when `max_steps` is not not provided.\n"
                "This can occur when `train dataloader` is not available to correctly "
                "prepare the scheduler.\n"
                "Scheduler will not be instantiated !"
            )
            return None

        # Get iters_per_batch
        max_epochs = scheduler_config.get('t_max_epochs')
        accumulate_grad_batches = scheduler_config.get('t_accumulate_grad_batches')
        limit_train_batches = scheduler_config.get('t_limit_train_batches')
        num_workers = scheduler_config.get('t_num_workers')

        # Compute effective num max_steps
        num_samples = len(train_dataloader.dataset)
        # TODO: not sure if this will be the correct LR schedule for Megatron
        # we may need to override ModelPT setup_optimization
        if train_dataloader.batch_size is not None:
            batch_size = train_dataloader.batch_size
        elif hasattr(train_dataloader, 'batch_sampler') and train_dataloader.batch_sampler is not None:
            if train_dataloader.batch_sampler.micro_batch_size is not None:
                batch_size = train_dataloader.batch_sampler.micro_batch_size
            else:
                raise ValueError(f'Could not find batch_size from batch_sampler: {train_dataloader.batch_sampler}')
        elif hasattr(train_dataloader, 'sampler') and train_dataloader.sampler is not None:
            if (
                hasattr(train_dataloader.sampler, 'micro_batch_size')
                and train_dataloader.sampler.micro_batch_size is not None
            ):
                batch_size = train_dataloader.sampler.micro_batch_size
            else:
                raise ValueError(f'Could not find batch_size from sampler: {train_dataloader.sampler}')
        else:
            raise ValueError(f'Could not find batch_size from train_dataloader: {train_dataloader}')
        drop_last = train_dataloader.drop_last

        max_steps = compute_max_steps(
            max_epochs=max_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_train_batches=limit_train_batches,
            num_workers=num_workers,
            num_samples=num_samples,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    else:
        logging.warning(
            "Neither `max_steps` nor `iters_per_batch` were provided to `optim.sched`, "
            "cannot compute effective `max_steps` !\n"
            "Scheduler will not be instantiated !"
        )
        return None

    # Inject max_steps (effective or provided) into the scheduler config
    scheduler_args['max_steps'] = max_steps

    # Get the scheduler class from the config
    scheduler_cls = get_scheduler(scheduler_name, **scheduler_args)

    # Pop 'max_steps' if it's not required by the scheduler
    if 'max_steps' not in inspect.signature(scheduler_cls).parameters:
        scheduler_args.pop('max_steps')

    # Instantiate the LR schedule
    schedule = scheduler_cls(optimizer, **scheduler_args)

    logging.info(
        'Scheduler "%s" \nwill be used during training (effective maximum steps = %d) - \nParameters : \n(%s)',
        str(schedule),
        max_steps,
        OmegaConf.to_yaml(OmegaConf.create(scheduler_args)),
    )

    # Wrap the schedule in PTL arguments to perform stepwise computation
    # Rather than epoch level computation
    if isinstance(schedule, optim.lr_scheduler.ReduceLROnPlateau):
        reduce_lr_on_plateau = True
    else:
        reduce_lr_on_plateau = False

    schedule_dict = {
        'scheduler': schedule,
        'interval': interval,
        'frequency': 1,
        'monitor': monitor,
        'reduce_on_plateau': reduce_lr_on_plateau,
    }
    return schedule_dict


def compute_max_steps(
    max_epochs, accumulate_grad_batches, limit_train_batches, num_workers, num_samples, batch_size, drop_last
):
    _round = math.floor if drop_last else math.ceil

    sampler_num_samples = math.ceil(num_samples / max(1, num_workers))

    if drop_last and num_workers > 1:
        logging.warning(
            "Please note that drop_last is broken in pytorch 1.6.0. We will fix when pytorch 1.7.0 is released"
        )
        # TODO: Master version, not in pytorch 1.6.0
        # sampler_num_samples = math.ceil((num_samples - num_workers)/ num_workers)

    steps_per_epoch = _round(sampler_num_samples / batch_size)
    if isinstance(limit_train_batches, int) or limit_train_batches == 0.0:
        steps_per_epoch = min(steps_per_epoch, int(limit_train_batches))
    elif steps_per_epoch != float('inf'):
        # limit_train_batches is a percentage of batches per epoch
        steps_per_epoch = int(steps_per_epoch * limit_train_batches)

    return math.ceil(steps_per_epoch / accumulate_grad_batches) * max_epochs


AVAILABLE_SCHEDULERS = {
    'WarmupPolicy': WarmupPolicy,
    'WarmupHoldPolicy': WarmupHoldPolicy,
    'SquareAnnealing': SquareAnnealing,
    'CosineAnnealing': CosineAnnealing,
    'NoamAnnealing': NoamAnnealing,
    'NoamHoldAnnealing': NoamHoldAnnealing,
    'WarmupAnnealing': WarmupAnnealing,
    'InverseSquareRootAnnealing': InverseSquareRootAnnealing,
    'T5InverseSquareRootAnnealing': T5InverseSquareRootAnnealing,
    'SquareRootAnnealing': SquareRootAnnealing,
    'PolynomialDecayAnnealing': PolynomialDecayAnnealing,
    'PolynomialHoldDecayAnnealing': PolynomialHoldDecayAnnealing,
    'StepLR': pt_scheduler.StepLR,
    'ExponentialLR': pt_scheduler.ExponentialLR,
    'ReduceLROnPlateau': pt_scheduler.ReduceLROnPlateau,
    'CyclicLR': pt_scheduler.CyclicLR,
}

EPOCH_SCHEDULERS = {
    'ExponentialLR': pt_scheduler.ExponentialLR,
    'ReduceLROnPlateau': pt_scheduler.ReduceLROnPlateau,
}
