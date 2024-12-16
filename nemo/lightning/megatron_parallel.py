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

import abc
import collections.abc
import functools
import inspect
import queue
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import torch
import torch.distributed
from lightning.pytorch.utilities import move_data_to_device
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as McoreDDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor, nn
from typing_extensions import override

DataT = TypeVar("DataT", Tensor, Dict[str, Tensor], Sequence[Tensor])
ModelT = TypeVar("ModelT", bound=nn.Module)
T = TypeVar('T')
STEP_OUTPUT = Optional[Union[Tensor, Mapping[str, Any]]]

if TYPE_CHECKING:
    import lightning.pytorch as pl


@runtime_checkable
class PrecisionPluginProtocol(Protocol[DataT]):
    def convert_input(self, data: DataT) -> DataT: ...

    def convert_output(self, output: torch.Tensor) -> torch.Tensor: ...


def default_data_step(dataloader_iter: Iterator[DataT]) -> DataT:
    """
    Moves the data to a device.

    In this case we unpack the dataloader iterator. There may be a wrapper on the dataloader
    iter from here: https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/fabric/strategies.py#L441.

    This will not subset the data for your with context parallel so please override this function if you
    want to use context parallel.

    Examples:
        If the dataloader_iter returns: [Tuple[<tensor>, <int>, <int>]] -> move to device
        If the dataloader_iter returns: [<tensor>, <tensor>] -> move to device

    Returns:
        DataT: The data moved to the device.
    """
    if parallel_state.get_context_parallel_world_size() > 1:
        raise ValueError(
            "Default data step is being used in a context parallel environment."
            "Please define your own data step that appropriately slices the data for context parallel."
        )

    batch = next(dataloader_iter)

    # If its wrapped in a tuple, unpack it.
    if isinstance(batch, tuple) and len(batch) == 3:
        batch = batch[0]

    return move_data_to_device(batch, torch.cuda.current_device())


def default_forward_step(model: nn.Module, batch, *args, **kwargs) -> torch.Tensor:
    return model(batch, *args, **kwargs)


def extract_ddp_funcs(ddp_config, pipeline):
    no_sync_func, grad_sync_func = None, None

    if getattr(ddp_config, "overlap_grad_reduce", False):
        no_sync_func = [model_chunk.no_sync for model_chunk in pipeline]
        no_sync_func = no_sync_func[0] if len(pipeline) == 1 else no_sync_func
        if getattr(ddp_config, "align_grad_reduce", False):
            grad_sync_func = [model_chunk.start_grad_sync for model_chunk in pipeline]
            grad_sync_func = grad_sync_func[0] if len(pipeline) == 1 else grad_sync_func

    return no_sync_func, grad_sync_func


class MegatronParallel(nn.ModuleList, Generic[ModelT]):
    """Implements distributed model parallelism that is based on Megatron-LM.

    This supports various forms of parallelism:
    - tensor-parallelism
    - pipeline-parallelism
    - virtual pipeline parallelism
    - expert parallelism
    - sequence parallelism

    Attributes
    ----------
        pipeline (Union[nn.Module, Iterable[nn.Module]]): The sequence of modules that
            constitute the pipeline.
        precision_plugin (Optional[PrecisionPluginProtocol]): An optional plugin for
            managing precision-specific operations.
        callbacks (CallbackConnector): A connector for managing and invoking callbacks.
        data_step (Callable[[Iterator[DataT]], DataT]): A function that takes an iterator
            over the data and returns the next batch.
        forward_step (Callable[[nn.Module, DataT], Tensor]): A function that defines the
            forward pass of a model.
        loss_reduction (Optional[Callable[[nn.Module], MegatronLossReduction]]): An optional
            function that defines how the loss is reduced.
        vp_size (Optional[int]): Virtual pipeline parallel size.
        ddp_config (Optional[DistributedDataParallelConfig]): An instance of Megatron core's
            DistributedDataParallelConfig which controls the Megatron DDP configuration.
        cpu (bool): Whether model should reside on CPU.
        convert_module_fn (Optional[Callable[[ModelT], nn.Module]]): An optional function to
            apply to the model parameters after initialization.

    Examples
    --------
        >>> from torch import nn
        >>> from megatron_ext.megatron_parallel import MegatronParallel
        >>> model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
        >>> megatron_model = MegatronParallel(model)
        >>> print(megatron_model)
        MegatronParallel(
          (0): Linear(in_features=10, out_features=10, bias=True)
          (1): ReLU()
          (2): Linear(in_features=10, out_features=5, bias=True)
        )

    References
    ----------
        Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019).
        Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.
        arXiv preprint arXiv:1909.08053.
    """

    def __init__(
        self,
        pipeline: Union[ModelT, Iterable[ModelT]],
        precision_plugin: Optional[PrecisionPluginProtocol] = None,
        callbacks: Optional["CallbackConnector"] = None,
        data_step: Optional[Callable[[Iterator[DataT]], DataT]] = None,
        forward_step: Optional[Callable[[ModelT, DataT], Tensor]] = None,
        loss_reduction: Optional[Callable[[ModelT], "MegatronLossReduction"]] = None,
        vp_size: Optional[int] = None,
        ddp_config: Optional[DistributedDataParallelConfig] = None,
        cpu: bool = False,
        convert_module_fn: Optional[Callable[[ModelT], nn.Module]] = None,
    ) -> None:
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes

        _pipeline: List[nn.Module]
        if isinstance(pipeline, nn.ModuleList):
            _pipeline = list(pipeline)
        elif isinstance(pipeline, nn.Module):
            _pipeline = [pipeline]
        else:
            _pipeline = pipeline

        if vp_size is not None:
            if len(_pipeline) == 1 and parallel_state.get_pipeline_model_parallel_world_size() > 1:
                from nemo.lightning import io

                parallel_state.set_virtual_pipeline_model_parallel_world_size(vp_size)
                for i in range(1, vp_size):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    _model = io.reinit(_pipeline[0])
                    if hasattr(_model, "configure_model"):
                        _model.configure_model()
                    _pipeline.append(_model)

        super().__init__(_pipeline)
        self.precision_plugin = precision_plugin
        self._cpu = cpu
        self.callbacks = callbacks or CallbackConnector()
        self.data_step = data_step or default_data_step
        self.forward_step = forward_step or default_forward_step
        self.loss_reduction: MegatronLossReduction = loss_reduction
        self.ddp_config = ddp_config
        self.convert_module_fn = convert_module_fn

    def forward(
        self,
        data: Union[DataT, Iterator[DataT], List[Iterator[DataT]]],
        forward_only: bool = True,
        data_step: Optional[Callable[[Iterator[DataT]], DataT]] = None,
        forward_step: Optional[Callable[[ModelT, DataT], Tensor]] = None,
        loss_reduction: Optional["MegatronLossReduction[DataT, Any]"] = None,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        step_i: Optional[int] = None,
        wrap_forward_step: bool = True,
    ) -> torch.Tensor:
        """The method performs the forward pass of the model.

        This method is responsible for executing the forward pass of the model. If `forward_only` is set to False,

        During the execution, it invokes various callbacks at different stages of the operation.
        For more info about that see [CallbackConnector].

        Args:
            data (Union[DataT, Iterator[DataT], List[Iterator[DataT]]]): The input data for the model.
            forward_only (bool, optional): If True, only perform the forward pass. Defaults to True.
            data_step (Optional[Callable[[Iterator[DataT]], DataT]], optional): Function to process the data. Defaults to None.
            forward_step (Optional[Callable[[nn.Module, DataT], Tensor]], optional): Function to perform the forward pass. Defaults to None.
            loss_reduction (Optional[MegatronLossReduction[DataT, Any]], optional): Function to reduce the loss. Defaults to None.
            seq_length (Optional[int], optional): Sequence length for the model. Defaults to None.
            micro_batch_size (Optional[int], optional): Size of the micro batch. Defaults to None.
            num_microbatches (Optional[int], optional): Number of microbatches. Defaults to None.
            wrap_forward_step (bool, optional): If True, wrap the forward step function. Defaults to True.

        Returns
        -------
            torch.Tensor: The output tensor from the forward pass.
        """
        _forward_step = forward_step or self.forward_step
        _loss_reduction = loss_reduction or self.loss_reduction
        _forward_context = {}

        if wrap_forward_step:
            _data_step = data_step or self.data_step
            forward_step_func = self.wrapped_forward_step(
                forward_step=_forward_step,
                data_step=_data_step,
                loss_reduction=_loss_reduction,
                context=_forward_context,
            )
        else:
            forward_step_func = _forward_step

        step = MegatronStep.infer(
            self,
            data,
            forward_step_func,
            forward_only=forward_only,
            micro_batch_size=micro_batch_size,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            step_i=step_i,
        )
        _forward_context["step"] = step
        step = self.callbacks.transform_event("on_megatron_step_start", step)

        self.callbacks.event("on_megatron_microbatches_start", step=step)
        microbatch_outputs = step()
        self.callbacks.event("on_megatron_microbatches_end", step=step, microbatch_outputs=microbatch_outputs)

        if microbatch_outputs:
            self.callbacks.event(
                "on_megatron_reduce_microbatches_start", step=step, microbatch_outputs=microbatch_outputs
            )

            if isinstance(_loss_reduction, _ModuleStepFunction):
                _loss_reduction = _loss_reduction(self[0])

            reduced = _loss_reduction.reduce(microbatch_outputs)
            self.callbacks.event(
                "on_megatron_reduce_microbatches_end",
                step=step,
                loss_reduction=_loss_reduction,
                microbatch_outputs=microbatch_outputs,
                reduced=reduced,
            )
        else:
            # we're not on the last pipeline stage so no losses
            reduced = torch.tensor(0.0, device=torch.cuda.current_device())

        self.callbacks.event("on_megatron_step_end", step=step, microbatch_outputs=microbatch_outputs, reduced=reduced)

        return reduced

    def training_step(
        self,
        data: DataT,
        data_step: Optional[Callable[[Iterator[DataT]], DataT]] = None,
        forward_step: Optional[Callable[[ModelT, DataT], Tensor]] = None,
        loss_reduction: Optional["MegatronLossReduction[DataT, Any]"] = None,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        **kwargs,
    ) -> STEP_OUTPUT:
        return self._step(
            "training",
            data,
            data_step=data_step,
            forward_step=forward_step,
            loss_reduction=loss_reduction,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            num_microbatches=num_microbatches,
            forward_only=False,
            **kwargs,
        )

    def validation_step(
        self,
        data: DataT,
        data_step: Optional[Callable[[Iterator[DataT]], DataT]] = None,
        forward_step: Optional[Callable[[ModelT, DataT], Tensor]] = None,
        loss_reduction: Optional["MegatronLossReduction[DataT, Any]"] = None,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        step_i: Optional[int] = None,
        **kwargs,
    ) -> STEP_OUTPUT:
        return self._step(
            "validation",
            data,
            data_step=data_step,
            forward_step=forward_step,
            loss_reduction=loss_reduction,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            num_microbatches=num_microbatches,
            step_i=step_i,
            forward_only=True,
            **kwargs,
        )

    def test_step(
        self,
        data: DataT,
        data_step: Optional[Callable[[Iterator[DataT]], DataT]] = None,
        forward_step: Optional[Callable[[ModelT, DataT], Tensor]] = None,
        loss_reduction: Optional["MegatronLossReduction[DataT, Any]"] = None,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        step_i: Optional[int] = None,
        **kwargs,
    ) -> STEP_OUTPUT:
        return self._step(
            "test",
            data,
            data_step=data_step,
            forward_step=forward_step,
            loss_reduction=loss_reduction,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            num_microbatches=num_microbatches,
            step_i=step_i,
            forward_only=True,
            **kwargs,
        )

    def predict_step(
        self,
        data: DataT,
        data_step: Optional[Callable[[Iterator[DataT]], DataT]] = None,
        forward_step: Optional[Callable[[ModelT, DataT], Tensor]] = None,
        loss_reduction: Optional["MegatronLossReduction[DataT, Any]"] = None,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        step_i: Optional[int] = None,
        **kwargs,
    ) -> STEP_OUTPUT:
        return self._step(
            "predict",
            data,
            data_step=data_step,
            forward_step=forward_step,
            loss_reduction=loss_reduction,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            num_microbatches=num_microbatches,
            step_i=step_i,
            forward_only=True,
            **kwargs,
        )

    def _step(
        self,
        step_type: str,
        data: DataT,
        data_step: Optional[Callable[[Iterator[DataT]], DataT]] = None,
        forward_step: Optional[Callable[[ModelT, DataT], Tensor]] = None,
        loss_reduction: Optional["MegatronLossReduction[DataT, Any]"] = None,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        forward_only: bool = True,
        step_i: Optional[int] = None,
        **kwargs,
    ) -> STEP_OUTPUT:
        if not hasattr(self.module, f"{step_type}_step"):
            raise AttributeError(f"self.module must have a `{step_type}_step` method")

        _data_step = data_step or _ModuleStepFunction.from_data_step(self.module, step_type)
        _forward_step = forward_step or _ModuleStepFunction.from_forward_step(self.module, step_type)
        _loss_reduction = loss_reduction or _ModuleStepFunction.from_loss_reduction(self.module, step_type)

        return self.forward(
            data=data,
            data_step=_data_step,
            forward_step=_forward_step,
            loss_reduction=_loss_reduction,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            step_i=step_i,
            **kwargs,
        )

    def wrapped_forward_step(
        self, forward_step, loss_reduction, data_step, context
    ) -> Callable[[nn.Module, DataT], Tuple[torch.Tensor, "MegatronCallbackProtocol"]]:
        """The method wraps the forward step function and returns a callable.

        The output is a forward_step function in the form of:
        https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L129

        Args:
            forward_step (Callable): The forward step function to be wrapped.
            loss_reduction (Callable): The loss reduction function.
            context (Dict): The context dictionary.
            data_step (Callable): The data step function.

        Returns
        -------
            Callable: The wrapped forward step function.
        """
        from megatron.core import parallel_state

        @functools.wraps(forward_step)
        def wrapped_forward_step_func(dataloader_iter, model):
            if isinstance(data_step, _ModuleStepFunction):
                _data_step = data_step(model)
            else:
                _data_step = data_step

            batch = _data_step(dataloader_iter)
            step = context["step"]

            if isinstance(loss_reduction, _ModuleStepFunction):
                forward_callback = loss_reduction(model)
            else:
                forward_callback = loss_reduction

            if isinstance(forward_step, _ModuleStepFunction):
                _forward_step = forward_step(model)
            else:
                _forward_step = forward_step

            self.callbacks.event(
                "on_megatron_microbatch_start",
                step=step,
                batch=batch,
                forward_callback=forward_callback,
            )

            if self.precision_plugin and parallel_state.is_pipeline_first_stage():
                batch = self.precision_plugin.convert_input(batch)

            output_tensor = _forward_step(model, batch)

            # callback
            self._setup_module(
                forward_callback,
                batch=batch,
                model=self,
                forward_module=model,
                tensor=output_tensor,
            )

            if self.precision_plugin and parallel_state.is_pipeline_last_stage():
                output_tensor = self.precision_plugin.convert_output(output_tensor)

            self.callbacks.event(
                "on_megatron_microbatch_end",
                step=step,
                batch=batch,
                output=output_tensor,
                forward_callback=forward_callback,
            )

            return output_tensor, forward_callback

        return wrapped_forward_step_func

    def init_model_parallel(self):
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes

        for model_module in self:
            if not self._cpu:
                model_module.cuda(torch.cuda.current_device())

            for param in model_module.parameters():
                set_defaults_if_not_set_tensor_model_parallel_attributes(param)

            if hasattr(model_module, "configure_model"):
                if not hasattr(model_module, "set_input_tensor"):
                    if hasattr(model_module.module, "set_input_tensor"):
                        model_module.set_input_tensor = model_module.module.set_input_tensor
                    else:
                        # TODO: What to do here?
                        pass

            # Print number of parameters.
            if parallel_state.model_parallel_is_initialized() and parallel_state.get_data_parallel_rank() == 0:
                from nemo.utils import logging

                num_params = _calc_number_of_params(list(self))
                num_trainable_params = _calc_number_of_trainable_params(list(self))

                msg = (
                    f" > number of parameters on (tensor, pipeline) model parallel rank "
                    f"({parallel_state.get_tensor_model_parallel_rank()}, {parallel_state.get_pipeline_model_parallel_rank()}): "
                    f"{num_params}"
                )
                logging.info(msg)

                if num_params != num_trainable_params:
                    logging.info(
                        f" > number of trainable parameters: {num_trainable_params} ({num_trainable_params / num_params:.2%} of total)"
                    )

        if self.convert_module_fn:
            self.apply_convert_module_fn()

        self.init_ddp()

    def apply_convert_module_fn(self):
        for i in range(len(self)):
            self[i] = self.convert_module_fn(self[i])

    def init_ddp(self):
        if not isinstance(self.ddp_config, DistributedDataParallelConfig):
            return

        from megatron.core import parallel_state

        for model_chunk_idx, model_chunk in enumerate(self):
            module = model_chunk.module

            # Mcore DistributedDataParallel has to be called with grad. Normally this call is redundant, but for
            # PEFT with num_sanity_val_steps > 0 this is necessary.
            init_ddp_context = nullcontext if all(x.requires_grad for x in module.parameters()) else torch.enable_grad

            # Turn off bucketing for model_chunk 2 onwards, since communication for these
            # model chunks is overlapped with compute anyway, or if using VP and overlapping
            # data parallel param gather with optimizer
            overlap_param_gather_with_optimizer_step = False
            if hasattr(self, "optim") and isinstance(self.optim.config, OptimizerConfig):
                overlap_param_gather_with_optimizer_step = self.optim.config.overlap_param_gather_with_optimizer_step
            disable_bucketing = (model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step

            with init_ddp_context():
                ddp = DDP(
                    module.config,
                    self.ddp_config,
                    module,
                    disable_bucketing=disable_bucketing,
                )

            model_chunk.module = ddp
            model_chunk.buffers = ddp.buffers  # We need to do this explicitly since this is a attr pytorch uses
            model_chunk.__class__.__getattr__ = getattr_proxy  # type: ignore

        # param_sync_func is set in nemo.lightning.pytorch.optim.megatron
        no_sync_func, grad_sync_func = extract_ddp_funcs(self.ddp_config, self)
        for module in self:
            module.config.no_sync_func = no_sync_func
            module.config.grad_sync_func = grad_sync_func

    def _setup_module(self, function, **kwargs) -> None:
        if hasattr(function, "setup"):
            setup_args = inspect.getfullargspec(function.setup).args
            setup_kwargs = {k: v for k, v in kwargs.items() if k in setup_args}
            function.setup(**setup_kwargs)

    def _call_module(self, function, *args, **kwargs) -> torch.Tensor:
        self._setup_module(function, **kwargs)

        call_args = inspect.getfullargspec(function).args
        call_kwargs = {k: v for k, v in kwargs.items() if k in call_args}
        output_tensor = function(*args, **call_kwargs)

        return output_tensor

    def sharded_state_dict(self, prefix: str = "") -> Dict[str, Any]:
        from megatron.core import parallel_state

        """
        Creates the sharded state dict which is used by dist_checkpoint to save the sharded tensors to disk.
        When given the sharded_stated_dict, dist_checkpoint.load will load the tensors corresponding to
        self.state_dict().
        The sharded tensor mapping is defined in the GPTModel class from mcore.
        """
        sharded_state_dict = {}
        for index, module in enumerate(self):
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                # virtual pipline rank must be set so that GPTModel returns the correct sharded state dict
                parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                module_sharded_state_dict = self._module_sharded_state_dict(module)
                sharded_state_dict[f"model_{index}"] = module_sharded_state_dict
            else:
                module_sharded_state_dict = self._module_sharded_state_dict(module)
                sharded_state_dict.update(module_sharded_state_dict)

        # reset vp rank
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

        return sharded_state_dict

    def _module_sharded_state_dict(self, module, *args, **kwargs) -> Dict[str, Any]:
        if hasattr(module, "sharded_state_dict"):
            return module.sharded_state_dict(*args, **kwargs)
        elif hasattr(module, "configure_model"):
            prefix = "".join([kwargs.pop("prefix", ""), "module."])
            return self._module_sharded_state_dict(module.module, *args, prefix=prefix, **kwargs)

        raise ValueError("Could not find sharded state dict")

    def enable_forward_pre_hook(self):
        for model in self:
            model_chunk = model.module
            assert isinstance(model_chunk, DDP)
            model_chunk.enable_forward_pre_hook()

    def disable_forward_pre_hook(self):
        for model in self:
            model_chunk = model.module
            assert isinstance(model_chunk, DDP)
            model_chunk.disable_forward_pre_hook()

    def force_param_sync(self):
        for model in self:
            model_chunk = model.module
            assert isinstance(model_chunk, DDP)
            model_chunk.start_param_sync(force_sync=True)

    @property
    def pipeline(self) -> Union[ModelT, List[ModelT]]:
        if len(self) == 1:
            return self[0]
        else:
            return list(self)

    @property
    def module(self) -> ModelT:
        return self[0]

    @override
    def __getattr__(self, item: Any) -> Any:
        try:
            # First, try to get the attribute from the superclass (nn.ModuleList)
            return super().__getattr__(item)
        except AttributeError:
            # If not found in superclass, check if we have any modules
            if len(self) == 0:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{item}' and contains no modules"
                )

            # Try to get it from the first module
            try:
                return getattr(self._modules[self._get_abs_string_index(0)], item)
            except AttributeError:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


class _ModuleStepFunction:
    """
    This class acts as a bridge between Megatron core's lower-level functional API and PTL's object-oriented API,
        making it possible to use PTL-compatible functions in Megatron core.
    """

    def __init__(self, name: str, is_property: bool = False, includes_self: bool = False):
        self.name = name
        self.is_property = is_property
        self.includes_self = includes_self

    @classmethod
    def from_data_step(cls, module: "pl.LightningModule", step_type: str) -> Optional["_ModuleStepFunction"]:
        for fn_name in [f"{step_type}_data_step", "data_step"]:
            if hasattr(module, fn_name):
                return _ModuleStepFunction(fn_name)

        return None

    @classmethod
    def from_forward_step(cls, module: "pl.LightningModule", step_type: str) -> Optional["_ModuleStepFunction"]:
        from megatron.core import parallel_state

        if parallel_state.is_pipeline_last_stage():
            if not hasattr(module, f"{step_type}_step"):
                raise ValueError(f"LightningModule does not have {step_type}_step method")

            return _ModuleStepFunction(f"{step_type}_step", includes_self=True)

        for fn_name in [f"{step_type}_forward_step", "forward_step"]:
            if hasattr(module, fn_name):
                return _ModuleStepFunction(fn_name, includes_self=True)

        return None

    @classmethod
    def from_loss_reduction(cls, module: "pl.LightningModule", step_type: str) -> Optional["_ModuleStepFunction"]:
        for fn_name in [f"{step_type}_loss_reduction", "loss_reduction"]:
            if hasattr(module, fn_name):
                return _ModuleStepFunction(fn_name, is_property=True)

        return None

    def __call__(self, module: nn.Module):

        attr = getattr(module, self.name)

        if self.is_property:
            if isinstance(getattr(type(module), self.name), property):
                return attr
            else:
                return attr()

        if self.includes_self:

            def wrapped(self, *args):
                return attr(*args)

            return wrapped

        return attr


def getattr_proxy(self, item: Any) -> Any:
    try:
        return super(self.__class__, self).__getattr__(item)
    except AttributeError as e:
        if item == 'module':  ## this is a hacky WAR and may cause misleading error messages
            raise e
        try:
            return getattr(self.module, item)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


class DDP(McoreDDP):
    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        **kwargs,
    ):
        init_parameters = inspect.signature(McoreDDP.__init__).parameters
        # Updates to the McoreDDP class have removed some parameters, so we need to
        #  filter out any kwargs that are not part of the updated signature, if a new
        #  version of mcore is being used.
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_parameters}
        super().__init__(
            config=config,
            ddp_config=ddp_config,
            module=module,
            disable_bucketing=disable_bucketing,
            **filtered_kwargs,
        )

    def state_dict(self, prefix='', keep_vars=False, **kwargs):
        self.module.state_dict(prefix=prefix, keep_vars=keep_vars, **kwargs)

    def __getattr__(self, item: Any) -> Any:
        return getattr_proxy(self, item)


class CallbackConnector:
    """
    A connector for managing and invoking callbacks.

    The CallbackConnector class in the MegatronParallel module
    is used to manage and invoke callbacks during the execution of the model.
    Callbacks are functions that are called at specific stages of the model
    execution, allowing you to hook into the model's operation for logging, debugging, or other purposes.

    The CallbackMethods class defines the names of the callback methods that can be used.

    These methods are:
    - `on_megatron_step_start`
    - `on_megatron_microbatch_start`
    - `on_megatron_microbatch_callback`
    - `on_megatron_microbatch_end`
    - `on_megatron_reduce_microbatches_start`
    - `on_megatron_reduce_microbatches_end`
    - `on_megatron_log_step_end`
    - `on_megatron_step_end`

    Each of these methods corresponds to a specific stage in the model's operation.
    You can define these methods in your callback functions to perform specific actions at these stages.
    There is no need for the class to be a subclass of a specific parent class. As long as the class contains the methods outlined above,
    it can be used as a callback.
    """

    def __init__(self, callbacks=None) -> None:
        self.callbacks = defaultdict(list)
        if callbacks:
            self.add(*callbacks)

    def add(self, *callbacks) -> "CallbackConnector":
        """
        Adds callback functions to the connector.

        Parameters
        ----------
        *callbacks : CallbackT
            One or more callback functions to add.

        Returns
        -------
        CallbackConnector
            The CallbackConnector instance to allow method chaining.
        """
        _pl_callback = None
        try:
            import lightning.pytorch as pl

            _pl_callback = pl.Callback
        except ImportError:
            pass

        megatron_methods = {m for m in dir(CallbackMethods) if m.startswith("on") and not hasattr(_pl_callback, m)}

        for callback in callbacks:
            if isinstance(callback, CallbackConnector):
                # Handle CallbackConnector instance: merge its callbacks
                for event_name, event_callbacks in callback.callbacks.items():
                    self.callbacks[event_name].extend(event_callbacks)
            else:
                for method in megatron_methods:
                    if hasattr(callback, method) and callable(getattr(callback, method)):
                        self.callbacks[method].append(callback)

        return self

    def event(self, name: str, *args, **kwargs) -> None:
        """
        Triggers an event and calls all associated callbacks.

        Parameters
        ----------
        name : str
            The name of the event to trigger.
        *args : Any
            Positional arguments to pass to the callbacks.
        **kwargs : Any
            Keyword arguments to pass to the callbacks.
        """
        for callback in self.callbacks.get(name, []):
            callback_method = getattr(callback, name, None)
            if callable(callback_method):
                # Inspect the callback method to determine accepted arguments
                sig = inspect.signature(callback_method)
                params = sig.parameters.values()

                # Check for *args and **kwargs in the callback method
                accepts_var_args = any(p.kind == p.VAR_POSITIONAL for p in params)
                accepts_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)

                if accepts_var_args and accepts_var_kwargs:
                    # If both *args and **kwargs are accepted, pass them directly
                    callback_method(*args, **kwargs)
                elif accepts_var_args:
                    # If only *args is accepted, filter kwargs
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    callback_method(*args, **filtered_kwargs)
                elif accepts_var_kwargs:
                    # If only **kwargs is accepted, filter args
                    filtered_args = [
                        arg
                        for arg, param in zip(args, params)
                        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                    ]
                    callback_method(*filtered_args, **kwargs)
                else:
                    # If neither is accepted, filter both args and kwargs
                    filtered_args = [
                        arg
                        for arg, param in zip(args, params)
                        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                    ]
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    callback_method(*filtered_args, **filtered_kwargs)

    def transform_event(self, name: str, obj: T, **kwargs) -> T:
        """
        Triggers an event that allows callbacks to transform and return an object.

        This method applies a series of potential transformations to the input object
        by calling registered callbacks. Each callback has the opportunity to modify
        and return a new version of the object.

        Parameters
        ----------
        name : str
            The name of the event to trigger.
        obj : T
            The object to be potentially transformed by callbacks.
        **kwargs : Any
            Additional keyword arguments to pass to the callbacks.

        Returns
        -------
        T
            The potentially transformed object.
        """
        for callback in self.callbacks.get(name, []):
            callback_method = getattr(callback, name, None)
            if callable(callback_method):
                result = callback_method(obj, **kwargs)

                # Update obj if the callback returned a value of the same type
                if result is not None and isinstance(result, type(obj)):
                    obj = result

        return obj

    def __add__(self, other) -> "CallbackConnector":
        """
        Adds another CallbackConnector's callbacks to this one.

        Parameters
        ----------
        other : CallbackConnector
            Another CallbackConnector instance to add.

        Returns
        -------
        CallbackConnector
            A new CallbackConnector instance with combined callbacks.

        Raises
        ------
        ValueError
            If `other` is not an instance of CallbackConnector.
        """
        if not isinstance(other, CallbackConnector):
            raise ValueError("Can only add CallbackConnector instances")
        new_connector = CallbackConnector()
        new_connector.callbacks = defaultdict(list, {**self.callbacks, **other.callbacks})
        return new_connector

    def __iadd__(self, other) -> "CallbackConnector":
        """
        In-place addition of another CallbackConnector's callbacks.

        Parameters
        ----------
        other : CallbackConnector
            Another CallbackConnector instance to add.

        Returns
        -------
        CallbackConnector
            The same CallbackConnector instance with combined callbacks.

        Raises
        ------
        ValueError
            If `other` is not an instance of CallbackConnector.
        """
        if not isinstance(other, CallbackConnector):
            raise ValueError("Can only add CallbackConnector instances")
        for event_name, event_callbacks in other.callbacks.items():
            self.callbacks[event_name].extend(event_callbacks)
        return self

    def __contains__(self, callback_object) -> bool:
        """
        Check if the given callback object is registered in the CallbackConnector.
        If the object has none of the methods of CallbackMethods, it returns True.
        If it has at least one of the methods, it checks if it's inside the CallbackConnector object.

        Args:
            callback_object: The object to check for callback methods.

        Returns
        -------
            bool: True if the callback object is registered, False otherwise.
        """
        # Get all method names from CallbackMethods class
        callback_methods = [
            func
            for func in dir(CallbackMethods)
            if callable(getattr(CallbackMethods, func)) and not func.startswith("__")
        ]

        # Check if the object has any method that's in CallbackMethods
        has_any_callback_method = any(hasattr(callback_object, method) for method in callback_methods)

        # If the object has none of the methods, it's not a callback
        if not has_any_callback_method:
            return True

        # If it has at least one of the methods, check if it's registered in the CallbackConnector
        for event_callbacks in self.callbacks.values():
            if callback_object in event_callbacks:
                return True

        return False


@dataclass
class MegatronStep(Generic[ModelT, DataT]):
    """
    Represents a single step in the Megatron model's training or inference process.

    This class encapsulates all the necessary information and logic for executing
    a single step (forward pass, and optionally backward pass) in the Megatron model.
    It handles data preparation, model execution, and provides utilities for inferring
    batch sizes and sequence lengths.

    Attributes:
        pipeline (MegatronParallel[ModelT]): The Megatron parallel model pipeline.
        data (Union[DataT, Iterator[DataT], List[Iterator[DataT]]]): Input data for the step.
        forward_step_func (Callable): Function to perform the forward step.
        forward_only (bool): If True, only perform forward pass (no backward pass).
        micro_batch_size (Optional[int]): Size of each micro-batch.
        seq_length (Optional[int]): Sequence length for the current step.
        num_microbatches (Optional[int]): Number of micro-batches in this step.
        decoder_seq_length (Optional[int]): Sequence length of decoder (used only in encoder-decoder style models) for the current step.

    Type Parameters:
        ModelT: The type of the model being used.
        DataT: The type of the input data.
    """

    pipeline: MegatronParallel[ModelT]
    data: Union[DataT, Iterator[DataT], List[Iterator[DataT]]]
    forward_step_func: Callable
    forward_only: bool
    micro_batch_size: Optional[int] = None
    seq_length: Optional[int] = None
    num_microbatches: Optional[int] = None
    step_i: Optional[int] = None
    decoder_seq_length: Optional[int] = None

    @classmethod
    def infer(
        cls,
        pipeline: MegatronParallel[ModelT],
        data: DataT,
        forward_step_func: Callable,
        forward_only: bool,
        micro_batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        step_i: Optional[int] = None,
    ) -> "MegatronStep[ModelT, DataT]":
        """
        Creates a MegatronStep instance, inferring missing parameters if possible.

        This method attempts to infer the micro_batch_size, seq_length, and num_microbatches
        from the provided data if they are not explicitly specified.

        Args:
            pipeline (MegatronParallel[ModelT]): The Megatron parallel model pipeline.
            data (DataT): Input data for the step.
            forward_step_func (Callable): Function to perform the forward step.
            forward_only (bool): If True, only perform forward pass (no backward pass).
            micro_batch_size (Optional[int]): Size of each micro-batch.
            seq_length (Optional[int]): Sequence length for the current step.
            num_microbatches (Optional[int]): Number of micro-batches in this step.
            step_i (Optional[int]): Step index for the current step.
        Returns:
            MegatronStep[ModelT, DataT]: An instance of MegatronStep with inferred parameters.
        """
        if step_i is None and pipeline.trainer:
            step_i = pipeline.trainer.global_step

        return cls(
            pipeline=pipeline,
            data=data,
            forward_step_func=forward_step_func,
            forward_only=forward_only,
            micro_batch_size=micro_batch_size or cls.infer_micro_batch_size(data),
            seq_length=seq_length or cls.infer_seq_length(data),
            num_microbatches=num_microbatches or cls.infer_num_microbatches(data),
            step_i=step_i,
        )

    def __call__(self) -> List[Any]:
        """
        Executes the Megatron step.

        This method performs the forward (and optionally backward) pass using the
        configured forward_backward_func. It ensures all necessary parameters are set
        before execution.

        Returns:
            List[Any]: The output of the forward_backward_func, typically containing
                       loss values and other relevant information.

        Raises:
            ValueError: If any of num_microbatches, seq_length, or micro_batch_size is not set.
        """
        if self.num_microbatches is None:
            raise ValueError("num_microbatches is not set")

        if self.seq_length is None:
            raise ValueError("seq_length is not set")

        if self.micro_batch_size is None:
            raise ValueError("micro_batch_size is not set")

        data_iterator, seq_length = self.get_data_iterator_and_seq_length()
        seq_length = seq_length or self.seq_length

        return self.forward_backward_func(
            forward_step_func=self.forward_step_func,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=self.num_microbatches,
            seq_length=seq_length,
            micro_batch_size=self.micro_batch_size,
            forward_only=self.forward_only,
            decoder_seq_length=self.decoder_seq_length,
        )

    def to_data_iterator_list(
        self, data: Union[DataT, Iterator[DataT], List[Iterator[DataT]]]
    ) -> List[Iterator[DataT]]:
        """
        Converts the provided data into a list of iterators.

        This method is used to convert the input data into a list of iterators that can be used
        for data parallelism in the Megatron model. The input data can be a single data item,
        an iterator, or a list of iterators.

        Args:
            data (Union[DataT, Iterator[DataT], List[Iterator[DataT]]]): The input data to be
                converted into a list of iterators.

        Returns:
            List[Iterator[DataT]]: A list of iterators created from the input data.
        """
        if isinstance(data, Iterator):
            return _make_data_iterator_list(self.model, data)
        elif isinstance(data, list) and all(isinstance(item, Iterator) for item in data):
            # If data is already a list of iterators, return it as is
            return cast(List[Iterator[DataT]], data)

        # For a single data item or any other type, wrap it in an iterator and return as a list
        return cast(List[Iterator[DataT]], [iter([data])])

    @classmethod
    def infer_micro_batch_size(cls, data: DataT) -> Optional[int]:
        """
        Infers the micro-batch size from the input data.

        This method attempts to determine the micro-batch size by examining the first
        dimension of the input data. It handles various data types including Tensors,
        dictionaries, lists, and tuples.

        Args:
            data (DataT): The input data from which to infer the micro-batch size.

        Returns:
            Optional[int]: The inferred micro-batch size, or None if it cannot be determined.
        """
        if isinstance(data, Tensor):
            return data.size(0)
        elif isinstance(data, dict):
            return cls.infer_micro_batch_size(next(iter(data.values())))
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            _tensor: Tensor = data[0]
            return cls.infer_micro_batch_size(_tensor)

        return None

    @classmethod
    def infer_seq_length(cls, data: DataT) -> Optional[int]:
        """
        Infers the sequence length from the input data.

        This method attempts to determine the sequence length by examining the second
        dimension of the input data. It handles various data types including Tensors,
        dictionaries, lists, and tuples.

        Args:
            data (DataT): The input data from which to infer the sequence length.

        Returns:
            Optional[int]: The inferred sequence length, or None if it cannot be determined.
        """
        if isinstance(data, Tensor):
            # TODO: Check if at least 2 dims
            return data.size(1)
        elif isinstance(data, dict):
            return cls.infer_seq_length(next(iter(data.values())))
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            _tensor: Tensor = data[0]
            return cls.infer_seq_length(_tensor)

        return None

    @classmethod
    def infer_num_microbatches(cls, data: DataT) -> Optional[int]:
        """
        Infers the number of micro-batches from the input data.

        Currently, this method assumes a single micro-batch for common data types.
        It may need to be extended for more complex data structures or use cases.

        Args:
            data (DataT): The input data from which to infer the number of micro-batches.

        Returns:
            Optional[int]: The inferred number of micro-batches, or None if it cannot be determined.
        """
        if isinstance(data, (dict, tuple, list, Tensor)):
            return 1

        return None

    @property
    def model(self) -> Union[ModelT, List[ModelT]]:
        """
        Retrieves the model or list of models from the pipeline.

        Returns:
            Union[ModelT, List[ModelT]]: The model or list of models in the pipeline.
        """
        return self.pipeline.pipeline

    @property
    def pl_module(self) -> "pl.LightningModule":
        """
        Retrieves the PyTorch Lightning module from the pipeline.

        Returns:
            pl.LightningModule: The PyTorch Lightning module.
        """
        return self.pipeline.module

    @property
    def trainer(self) -> "pl.Trainer":
        """
        Retrieves the PyTorch Lightning trainer from the pipeline.

        Returns:
            pl.Trainer: The PyTorch Lightning trainer.
        """
        return self.pipeline.trainer

    @functools.cached_property
    def forward_backward_func(self) -> "MegatronStepProtocol":
        """
        Retrieves the forward-backward function for the Megatron model.

        This property uses Megatron's scheduling to get the appropriate
        forward-backward function based on the current configuration.

        Returns:
            MegatronStepProtocol: The function to perform forward and backward passes.
        """
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        return get_forward_backward_func()

    def get_data_iterator_and_seq_length(self) -> Tuple[List[Iterator[DataT]], Optional[int]]:
        """
        Converts the provided data into a list of iterators.

        For finetuning, where sequence length is different for each step, this function also outputs the
        sequence length of the current batch.

        Returns:
            List[Iterator[DataT]]: A list of iterators created from the input data.
        """
        if self.has_global_batch_sampler:
            batch = next(self.data)
            if isinstance(batch, tuple) and len(batch) == 3:
                batch = batch[0]
            # finetuning can have dynamic sequence lengths
            seq_length = batch['tokens'].size(1) if 'tokens' in batch else None
            from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split

            data = get_iterator_k_split(batch, self.num_microbatches, True)
        else:
            data = self.data
            # for pretraining (fixed sequence length), we use seq_length inferred from the data sampler.
            seq_length = None
        return self.to_data_iterator_list(data), seq_length

    @functools.cached_property
    def has_global_batch_sampler(self) -> bool:
        # FIXME: cleanup the following code is here for backwards compatibility with nemo1.
        # The "batch" sampler is a nemo1 sampler. It requires some custom code here to use
        # (if use_global_batch_sampler), by default we shouldn't use this "batch" sampler probably.
        if getattr(self.trainer, "datamodule", None) is not None:
            use_global_batch_sampler = self.trainer.datamodule.data_sampler.dataloader_type == 'batch'
        elif getattr(self.trainer, "predict_dataloaders", None) is not None:
            from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (  # noqa: I001
                MegatronPretrainingBatchSampler,
            )

            # The batch_sampler gets injected into the dataloader by the data_sampler. When doing
            # predict without a datamodule we can look inside the dataloader's batch_sampler to see
            # if it is the nemo1 style sampler that we need to handle specially below.
            use_global_batch_sampler = isinstance(
                self.trainer.predict_dataloaders.batch_sampler, MegatronPretrainingBatchSampler
            )
        else:
            use_global_batch_sampler = False
        return use_global_batch_sampler


class CallbackMethods:
    """
    Defines callback methods for various stages of the Megatron model's execution.

    This class outlines the structure for callbacks that can be implemented to hook into
    different phases of the Megatron model's training or inference process. Each method
    represents a specific point in the execution where custom logic can be inserted.
    """

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """
        Called at the beginning of each Megatron step.

        This method is invoked before any processing of the step begins. It allows for
        any necessary setup or initialization for the step.

        Args:
            step (MegatronStep): The MegatronStep object representing the current step.

        Returns:
            MegatronStep: The potentially modified MegatronStep object.
        """
        ...

    def on_megatron_microbatches_start(self, step: MegatronStep) -> None:
        """
        Called before processing of microbatches begins.

        This method is invoked just before the model starts processing the microbatches
        within a step. It can be used for any preparations needed before microbatch processing.

        Args:
            step (MegatronStep): The MegatronStep object representing the current step.
        """
        ...

    def on_megatron_microbatch_start(
        self,
        step: MegatronStep,
        batch: DataT,
        forward_callback: "MegatronLossReduction",
    ) -> None:
        """
        Called at the start of processing each microbatch.

        This method is invoked before the forward pass of each microbatch. It provides
        access to the current batch data and the loss reduction callback.

        Args:
            step (MegatronStep): The MegatronStep object representing the current step.
            batch (DataT): The current microbatch of data being processed.
            forward_callback (MegatronLossReduction): The callback for loss reduction.
        """
        ...

    def on_megatron_microbatch_end(
        self,
        step: MegatronStep,
        batch: DataT,
        forward_callback: "MegatronLossReduction",
        output: Any,
    ) -> None:
        """
        Called at the end of processing each microbatch.

        This method is invoked after the forward pass of each microbatch. It provides
        access to the processed batch, the loss reduction callback, and the output of the forward pass.

        Args:
            step (MegatronStep): The MegatronStep object representing the current step.
            batch (DataT): The microbatch of data that was processed.
            forward_callback (MegatronLossReduction): The callback for loss reduction.
            output (Any): The output from the forward pass for this microbatch.
        """
        ...

    def on_megatron_microbatches_end(self, step: MegatronStep, microbatch_outputs: List[Any]) -> None:
        """
        Called after all microbatches in a step have been processed.

        This method is invoked once all microbatches within a step have been processed.
        It provides access to the outputs from all microbatches.

        Args:
            step (MegatronStep): The MegatronStep object representing the current step.
            microbatch_outputs (List[Any]): A list of outputs from all processed microbatches.
        """
        ...

    def on_megatron_reduce_microbatches_start(
        self,
        step: MegatronStep,
        microbatch_outputs: List[Any],
    ) -> None:
        """
        Called before the reduction of microbatch outputs begins.

        This method is invoked just before the model starts reducing (e.g., averaging)
        the outputs from all microbatches. It can be used for any preparations needed
        before the reduction process.

        Args:
            step (MegatronStep): The MegatronStep object representing the current step.
            microbatch_outputs (List[Any]): A list of outputs from all processed microbatches.
        """
        ...

    def on_megatron_reduce_microbatches_end(
        self,
        step: MegatronStep,
        microbatch_outputs: List[Any],
        loss_reduction: "MegatronLossReduction",
        reduced: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> None:
        """
        Called after the reduction of microbatch outputs is complete.

        This method is invoked after the model has finished reducing the outputs from
        all microbatches. It provides access to the original microbatch outputs,
        the loss reduction object, and the final reduced output.

        Args:
            step (MegatronStep): The MegatronStep object representing the current step.
            microbatch_outputs (List[Any]): A list of outputs from all processed microbatches.
            loss_reduction (MegatronLossReduction): The object used for loss reduction.
            reduced (Union[torch.Tensor, Dict[str, torch.Tensor]]): The final reduced output.
        """
        ...

    def on_megatron_step_end(
        self,
        step: MegatronStep,
        microbatch_outputs: List[Any],
        reduced: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """
        Called at the end of each Megatron step.

        This method is invoked after all processing for a step is complete. It provides
        access to the outputs from all microbatches and the final reduced output (if available).

        Args:
            step (MegatronStep): The MegatronStep object representing the current step.
            microbatch_outputs (List[Any]): A list of outputs from all processed microbatches.
            reduced (Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]): The final reduced
                output, if available. This may be None for certain configurations or pipeline stages.
        """
        ...


ReductionT = TypeVar("ReductionT")


class MegatronLossReduction(nn.Module, Generic[DataT, ReductionT]):
    def __init__(self) -> None:
        super(MegatronLossReduction, self).__init__()
        self.batch = None
        self.register_forward_pre_hook(self._pre_forward_hook)

    def setup(self, batch) -> None:
        self.batch = batch

    def _pre_forward_hook(self, module, x):
        return (self.batch,) + x

    def forward(self, batch: DataT, forward_out: torch.Tensor) -> Tuple[torch.Tensor, ReductionT]:
        raise NotImplementedError("Must be implemented by subclass.")

    @abc.abstractmethod
    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass.")


@runtime_checkable
class MegatronCallbackProtocol(Protocol):
    def __call__(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: ...


@runtime_checkable
class MegatronStepProtocol(Protocol):
    def __call__(
        self,
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: Optional[int] = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
    ) -> list: ...


def _calc_number_of_params(model: List[nn.Module]) -> int:
    assert isinstance(model, list)

    return sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])


def _calc_number_of_trainable_params(model: List[nn.Module]) -> int:
    assert isinstance(model, list)

    return sum([sum([p.numel() for p in model_module.parameters() if p.requires_grad]) for model_module in model])


def is_list_of_iterators(var) -> bool:
    if not isinstance(var, list):
        return False

    return all(isinstance(item, collections.abc.Iterator) for item in var)


def _make_data_iterator_list(model, data_iterator: Iterator) -> List[Iterator]:
    """Convert data iterator into form expected by Megatron.

    With interleaved pipeline parallelism, Megatron expects a
    list of one data iterator per model chunk. Each model
    chunk independently gets data from its data iterator, so
    we need to interact with the data iterator multiple times
    for each microbatch step. Instead of incorporating this
    logic into the data loader, we cache the iterator's output
    to the first model chunk and reuse it in the other model
    chunks.
    """
    if not isinstance(model, list) or len(model) == 1:
        return data_iterator  # TODO @tmoon: Remove
        # TODO @tmoon: Use once available in Megatron-LM
        # return DataIteratorList([data_iterator])

    class CachingIterator:
        """Iterator wrapper that caches values."""

        class Proxy:
            """Returns values from caching iterator wrapper.

            Assumed to never advance past the caching iterator.
            """

            def __init__(self):
                self.cache = queue.Queue()

            def __iter__(self):
                return self

            def __next__(self):
                return self.cache.get_nowait()

        def __init__(self, iterator: Iterator):
            self.iterator = iterator
            self.proxies = []

        def make_proxy(self):
            self.proxies.append(CachingIterator.Proxy())
            return self.proxies[-1]

        def __iter__(self):
            return self

        def __next__(self):
            val = next(self.iterator)
            for proxy in self.proxies:
                proxy.cache.put(val)
            return val

    # Make list of iterator wrappers
    iters = [CachingIterator(data_iterator)]
    while len(iters) < len(model):
        iters.append(iters[0].make_proxy())
    return iters  # TODO @tmoon: Remove
    # TODO @tmoon: Use once available in Megatron-LM
    # return DataIteratorList(iters)


class MaskedTokenLossReduction(MegatronLossReduction):
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Taken from:
        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L951-L976 .
        """
        from megatron.core import parallel_state

        from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

        # neva returns (logits, loss_mask)
        if isinstance(forward_out, tuple):
            forward_out, loss_mask = forward_out
            batch["loss_mask"] = loss_mask
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            loss_for_ub = masked_token_loss(forward_out, batch["loss_mask"])
        else:
            loss_for_ub = masked_token_loss_context_parallel(
                forward_out, batch["loss_mask"], batch['num_valid_tokens_in_ub']
            )

        if self.validation_step and not self.val_drop_last:
            num_valid_tokens_in_ub = batch["loss_mask"].sum()
            if loss_for_ub.isnan():
                assert batch["loss_mask"].count_nonzero() == 0, "Got NaN loss with non-empty input"
                loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
            else:
                loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

            loss_sum_and_ub_size_all_gpu = torch.cat(
                [
                    loss_sum_for_ub.clone().detach().view(1),
                    torch.tensor([num_valid_tokens_in_ub], device=torch.cuda.current_device()).clone().detach(),
                ]
            )
            torch.distributed.all_reduce(loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group())
            return loss_for_ub * cp_size, {"loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu}

        reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
        return loss_for_ub * cp_size, {"avg": reduced_loss}

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """Taken from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L535-L552 ."""
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)

                return loss_tensor.mean()

            # Get the total loss since micro batches sizes are not uniform
            loss_sum_tensors_list: List[torch.Tensor] = [
                loss_sum["loss_sum_and_ub_size"]
                for loss_sum in losses_reduced_per_micro_batch
                if loss_sum["loss_sum_and_ub_size"][1] > 0
            ]
            loss_sum = (
                torch.vstack(loss_sum_tensors_list).sum(dim=0)
                if len(loss_sum_tensors_list) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            return loss_sum

        return torch.tensor(0.0, device=torch.cuda.current_device())


class MaskedTokenLossReductionWithLossMask(MaskedTokenLossReduction):
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        forward_out: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # expecting returns (token_level_loss, loss_mask)
        forward_out, loss_mask = forward_out
        batch["loss_mask"] = loss_mask

        return super().forward(batch, forward_out)


def masked_token_loss(tensor: Tensor, mask: Tensor):
    """
    The function takes as input per-token loss and masks non-required values.
    """
    losses = tensor.float()
    loss_mask = mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll

    return loss


def masked_token_loss_context_parallel(tensor: Tensor, mask: Tensor, num_valid_tokens_in_ub: int):
    """
    masked token loss for CP > 1 as a separate function for readability.
    """
    from megatron.core import parallel_state

    losses = tensor.float()
    loss_mask = mask.view(-1).float()
    if num_valid_tokens_in_ub is None:
        num_valid_tokens_in_ub = loss_mask.sum()
    if num_valid_tokens_in_ub < 0.5:  # no valid tokens
        num_valid_tokens_in_ub += 1.0
    loss = torch.sum(losses.view(-1) * loss_mask) / num_valid_tokens_in_ub  # sequence level nll
    torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())

    return loss


@contextmanager
def moe_loss_tracker_ctx():
    from megatron.core.transformer.moe.moe_utils import (
        clear_aux_losses_tracker,
        reduce_aux_losses_tracker_across_ranks,
    )

    reduce_aux_losses_tracker_across_ranks()
    try:
        yield
    finally:
        clear_aux_losses_tracker()


@torch.no_grad()
def aggregate_moe_loss_stats(loss_scale=1.0):
    with moe_loss_tracker_ctx():
        tracker = parallel_state.get_moe_layer_wise_logging_tracker()
        aux_losses = {k: v['values'].float() * loss_scale for k, v in tracker.items()}
        total_loss_dict = {}
        for name, loss_list in aux_losses.items():
            if name not in total_loss_dict:
                total_loss_dict[name] = 0
            total_loss_dict[name] += loss_list.mean().item()
        return total_loss_dict
