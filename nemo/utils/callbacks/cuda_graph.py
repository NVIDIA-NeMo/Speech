# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# CUDAGraphCallback is a full iteration CUDA graph callback designed for
# models with PyTorch Lightning first, this has been tested with Stable
# Diffusion right now.
#
# Prerequisites for this callback:
# 1. Capturable: user has to make sure (almost) all the host & device
#    synchronizations are removed, some of the syncs regarding logging
#    of metrics introduced by PyTorch Lightning itself have been removed
#    by this callback. This ensures the graph can be captured.
# 2. Topology: user has to make sure there's no dynamic control flow
#    within the iteration. Please use APEX alternatives for building
#    blocks that contain dynamic control flow, e.g. gradient clipping.
#    Otherwise the captured graph can run, but may raise silent failure,
#    e.g. NaN loss.
# 3. Parameters: user has to make sure pointers involved in the graph
#    capturing range don't change across iterations. In this case users
#    have to ensure that data is copied to static tensors. Otherwise this
#    can also lead to silent failure.

import os
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loops.optimization.automatic import ClosureResult
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection, _ResultMetric
from pytorch_lightning.utilities import CombinedLoader, rank_zero_info
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.parallel import DistributedDataParallel

__all__ = ["CUDAGraphCallback"]


def struct_copy_one(src):
    if isinstance(src, tuple):
        return tuple(struct_copy_one(i) for i in src)
    elif isinstance(src, list):
        return list(struct_copy_one(i) for i in src)
    elif isinstance(src, dict):
        return {k: struct_copy_one(src[k]) for k in src}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def struct_copy_two(tgt, src):
    if isinstance(src, tuple):
        raise Exception(f"Unsupported copy for tuple yet: {type(src)}")
    elif isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                struct_copy_two(tgt[i], src[i])
            else:
                tgt[i] = src[i]
    elif isinstance(src, dict):
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                struct_copy_two(tgt[k], src[k])
            else:
                tgt[k] = src[k]
    elif isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)
    else:
        raise Exception(f"Expect top-level as container type but got: {type(src)}")


class StaticBufferLoader:
    """Load data to static buffers."""

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.static = None

    def __iter__(self):
        for inputs in self.loader:
            if self.static is None:
                with torch.cuda.stream(self.stream):
                    self.static = struct_copy_one(inputs)

            with torch.cuda.stream(self.stream):
                struct_copy_two(self.static, inputs)
            torch.cuda.current_stream().wait_stream(self.stream)
            yield self.static

    def __len__(self):
        return len(self.loader)


def get_lr(lr_scheduler):
    lrs = lr_scheduler.__orig_get_lr__()
    if not hasattr(lr_scheduler, "static_lrs"):
        lr_scheduler.static_lrs = lrs
    for i in range(len(lrs)):
        lr_scheduler.static_lrs[i].copy_(lrs[i])
    return lr_scheduler.static_lrs


def zero_grad(optimizer, *args, **kwargs):
    # We invoke zero_grad before graph capturing.
    if torch.cuda.is_current_stream_capturing():
        rank_zero_info("CUDAGraphCallback: set optimizer.zero_grad as nop during graph capturing.")
    else:
        optimizer.__orig_zero_grad__(*args, **kwargs)


def to_tensor(self, value, name):
    # Log metrics in PyTorch Lightning often invokes CPU & GPU synchronizations. Here
    # we implement smart metrics to avoid those synchronizations.
    # Refer to: https://github.com/Lightning-AI/pytorch-lightning/blob/2.0.7/src/lightning/pytorch/core/module.py#L615
    value = value.clone().detach() if isinstance(value, torch.Tensor) else torch.tensor(value)
    if not torch.numel(value) == 1:
        raise ValueError(
            f"`self.log({name}, {value})` was called, but the tensor must have a single element."
            f" You can try doing `self.log({name}, {value}.mean())`"
        )
    value = value.squeeze()
    return value


def register_key(self, key, meta, value):
    # PyTorch Lightning creates all metrics on GPU, but creating the metric on
    # its input device is prefered.
    # Refer to: https://github.com/Lightning-AI/pytorch-lightning/blob/2.0.7/src/lightning/pytorch/trainer/connectors/logger_connector/result.py#L409
    metric = _ResultMetric(meta, isinstance(value, torch.Tensor))
    device = value.device if isinstance(value, torch.Tensor) else self.device
    metric = metric.to(device)
    self[key] = metric


def update_metrics(self, key, value, batch_size):
    # PyTorch Lightning always move all metrics to GPU, but moving the metric to
    # its input device is prefered.
    result_metric = self[key]
    device = value.device if isinstance(value, torch.Tensor) else self.device
    result_metric.forward(value.to(device), batch_size)
    result_metric.has_reset = False


def get_optimizer_step(state):
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None,
    ) -> None:
        # Not all optimizer supports set_to_none.
        if not hasattr(optimizer, "support_set_to_none"):
            optimizer.support_set_to_none = is_param_in_hook_signature(
                optimizer.zero_grad, "set_to_none", explicit=True
            )
        if optimizer.support_set_to_none:
            zero_grad_kwargs = {"set_to_none": True}
        else:
            zero_grad_kwargs = {}

        if 0 <= state.current_iteration < state.capture_iteration or state.capture_iteration < 0:
            state.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(state.stream):
                optimizer.zero_grad(**zero_grad_kwargs)
                self.__orig_optimizer_step__(
                    epoch,
                    batch_idx,
                    optimizer,
                    optimizer_closure=optimizer_closure,
                )
            torch.cuda.current_stream().wait_stream(state.stream)

        if state.current_iteration == state.capture_iteration:
            torch.cuda.synchronize()
            # Sleep for one second to let environment stable
            time.sleep(1)
            rank_zero_info("CUDAGraphCallback: capturing CUDA graph for module %s.", self.__class__.__name__)
            with torch.cuda.graph(state.graph, stream=state.stream, capture_error_mode="global"):
                # PyTorch CUDA graph doc for whole-network capturing mentions:
                #
                #   Sets grads to None before capture, so backward() will create
                #   .grad attributes with allocations from the graph's private pool
                #
                # But it's not necessary, and it can lead to CUDA kernels inside
                # `zero_grad()` being not captured.
                optimizer.zero_grad(**zero_grad_kwargs)
                self.__orig_optimizer_step__(
                    epoch,
                    batch_idx,
                    optimizer,
                    optimizer_closure=optimizer_closure,
                )
            torch.cuda.synchronize()

        # Graph replay and reconstruct missing result
        if state.current_iteration >= state.capture_iteration >= 0:
            state.graph.replay()
            optimizer_closure._result = ClosureResult.from_training_step_output(state.output)

        # If something is not capturable, try to put it there, e.g. `self.log()`.
        if hasattr(self, "non_cuda_graph_capturable"):
            self.non_cuda_graph_capturable()

        state.current_iteration += 1

    return optimizer_step


def get_training_step(state):
    def training_step(self, batch):
        results = self.__orig_training_step__(batch)
        if state.output is None:
            state.output = struct_copy_one(results)

        # Copy results to static buffer to rebuild states required by PL.
        with torch.no_grad():
            struct_copy_two(state.output, results)
        return results

    return training_step


def get_amp_autocast_init(state):
    def amp_autocast_init(self, *args, **kwargs):
        if "cache_enabled" not in kwargs:
            kwargs["cache_enabled"] = False
        if state.current_iteration == 0:
            rank_zero_info("CUDAGraphCallback: disable autocast cache.")
        return self.__orig_init__(*args, **kwargs)

    return amp_autocast_init


def get_ddp_init(state):
    def init(self, *args, **kwargs):
        rank_zero_info("CUDAGraphCallback: init DDP on side stream.")
        with torch.cuda.stream(state.stream):
            self.__orig_init__(*args, **kwargs)

    return init


@dataclass
class CUDAGraphState:
    current_iteration: int = 0
    capture_iteration: int = -1  # -1 to disable
    stream: torch.cuda.Stream = None
    graph: torch.cuda.CUDAGraph = None
    output: Any = None  # static forward output


class CUDAGraphCallback(Callback):
    """Full iteration CUDA graph callback.

    Dataloader and LR scheduler are not included in the CUDA graph with this callback.
    """

    def __init__(self, capture_iteration=-1):
        super().__init__()

        # Required by CUDA graph with DDP
        # Ref: https://pytorch.org/docs/stable/notes/cuda.html#usage-with-distributeddataparallel
        if 0 <= capture_iteration <= 11:
            raise Exception("Warmup must run at least 11 DDP-enabled eager iterations before capture.")
        if torch.distributed.is_initialized():
            raise Exception("CUDAGraphCallback should be initialized before process group.")
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        self.state = CUDAGraphState(capture_iteration=capture_iteration)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        if self.state.capture_iteration < 0:
            return

        # Hack to avoid CUDA graph issue with AMP, PyTorch Lightning doesn't support
        # changing autocast arguments for now.
        # https://github.com/pytorch/pytorch/blob/v1.13.1/torch/cuda/graphs.py#L234
        torch.autocast.__orig_init__ = torch.autocast.__init__
        torch.autocast.__init__ = get_amp_autocast_init(self.state)

        # Before full-backward capture, DDP must be constructed in a side-stream context.
        # We've merged the change that init DDP on side stream to PyTorch Lightning V2,
        # but not all user defined strategy init DDP on side stream.
        DistributedDataParallel.__orig_init__ = DistributedDataParallel.__init__
        DistributedDataParallel.__init__ = get_ddp_init(self.state)

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune ends."""
        if self.state.capture_iteration < 0:
            return

        torch.autocast.__init__ = torch.autocast.__orig_init__
        del torch.autocast.__orig_init__

        DistributedDataParallel.__init__ = DistributedDataParallel.__orig_init__
        del DistributedDataParallel.__orig_init__

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit begins."""
        if self.state.capture_iteration < 0:
            return

        if is_param_in_hook_signature(pl_module.training_step, "dataloader_iter", explicit=True):
            raise Exception(
                "Found `dataloader_iter` argument in the `training_step`. This is "
                "not supported by full iteration CUDA graph capturing yet since "
                "dataloader will be within the CUDA graph capturing range.\n"
                "Try to change `dataloader_iter` to `batch` and remove "
                "`next(dataloader_iter)` from `training_step`."
            )

        # Now that CUDA device has been set, we can init stream and graph now
        self.state.stream = torch.cuda.Stream()
        self.state.graph = torch.cuda.CUDAGraph()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends."""
        if self.state.capture_iteration < 0:
            return

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins."""
        if self.state.capture_iteration < 0:
            return

        # Ensure training dataloader loads data to static buffer
        dataloader = trainer.fit_loop._combined_loader._iterables
        assert isinstance(
            dataloader, torch.utils.data.dataloader.DataLoader
        ), f"Expect Dataloader type but got {type(dataloader)}"
        static_loader = StaticBufferLoader(dataloader)
        _mode = trainer.fit_loop._combined_loader._mode
        combined_loader = CombinedLoader(static_loader, mode=_mode)
        trainer.fit_loop.__orig_combined_loader__ = trainer.fit_loop._combined_loader
        trainer.fit_loop._combined_loader = combined_loader
        trainer.fit_loop._data_fetcher.setup(trainer.fit_loop._combined_loader)
        iter(trainer.fit_loop._data_fetcher)

        # Warn if `optimizer.zero_grad()` invoked during graph capturing
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, torch.optim.Optimizer), f"Expect Optimizer type but got {type(optimizer)}"
            optimizer.__orig_zero_grad__ = optimizer.zero_grad
            optimizer.zero_grad = MethodType(zero_grad, optimizer)

        # Ensure LR scheduler writes to static buffer
        # We don't include LR scheduler in the full CUDA graph for now since
        # its overhead is very small.
        for config in trainer.lr_scheduler_configs:
            assert isinstance(
                config.scheduler, torch.optim.lr_scheduler._LRScheduler
            ), f"Expect _LRScheduler type but got {type(config.scheduler)}"
            config.scheduler.__orig_get_lr__ = config.scheduler.get_lr
            config.scheduler.get_lr = MethodType(get_lr, config.scheduler)

        # Use smart metrics to avoid syncs
        LightningModule.__orig_to_tensor__ = LightningModule._LightningModule__to_tensor
        LightningModule._LightningModule__to_tensor = to_tensor
        _ResultCollection.__orig_register_key__ = _ResultCollection.register_key
        _ResultCollection.register_key = register_key
        _ResultCollection.__orig_update_metrics__ = _ResultCollection.update_metrics
        _ResultCollection.update_metrics = update_metrics

        # Save model outputs to static buffer for PL states reconstruct
        pl_module.__orig_training_step__ = pl_module.training_step
        training_step = get_training_step(self.state)
        pl_module.training_step = MethodType(training_step, pl_module)

        # Capture CUDA graph from model forward propagation to optimizer step
        pl_module.__orig_optimizer_step__ = pl_module.optimizer_step
        optimizer_step = get_optimizer_step(self.state)
        pl_module.optimizer_step = MethodType(optimizer_step, pl_module)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train ends."""
        if self.state.capture_iteration < 0:
            return

        trainer.fit_loop._combined_loader = trainer.fit_loop.__orig_combined_loader__
        trainer.fit_loop._data_fetcher.setup(trainer.fit_loop._combined_loader)
        iter(trainer.fit_loop._data_fetcher)
        del trainer.fit_loop.__orig_combined_loader__

        for optimizer in trainer.optimizers:
            optimizer.zero_grad = optimizer.__orig_zero_grad__
            del optimizer.__orig_zero_grad__

        for config in trainer.lr_scheduler_configs:
            config.scheduler.get_lr = config.scheduler.__orig_get_lr__
            del config.scheduler.__orig_get_lr__

        LightningModule._LightningModule__to_tensor = LightningModule.__orig_to_tensor__
        del LightningModule.__orig_to_tensor__
        _ResultCollection.register_key = _ResultCollection.__orig_register_key__
        del _ResultCollection.__orig_register_key__
        _ResultCollection.update_metrics = _ResultCollection.__orig_update_metrics__
        del _ResultCollection.__orig_update_metrics__

        pl_module.training_step = pl_module.__orig_training_step__
        del pl_module.__orig_training_step__

        pl_module.optimizer_step = pl_module.__orig_optimizer_step__
        del pl_module.__orig_optimizer_step__

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """
        pass

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
        pass

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        pass

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        r"""
        Called when saving a checkpoint to give you a chance to store anything else you might want to save.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.module.LightningModule` instance.
            checkpoint: the checkpoint dictionary that will be saved.
        """
        # Since we've add bound method to optimizer and lr_scheduler, it can lead to more
        # CUDA tensors passed to consumer process unexpectedly.
        if "optimizer_states" in checkpoint:
            for optimizer_state in checkpoint["optimizer_states"]:
                for k in list(optimizer_state.keys()):
                    v = optimizer_state[k]
                    if isinstance(v, MethodType) and hasattr(v, "__self__"):
                        del optimizer_state[k]
        if "lr_schedulers" in checkpoint:
            for lr_scheduler in checkpoint["lr_schedulers"]:
                for k in list(lr_scheduler.keys()):
                    v = lr_scheduler[k]
                    if isinstance(v, MethodType) and hasattr(v, "__self__"):
                        del lr_scheduler[k]
