# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import collections
import itertools
from typing import Callable, Iterable, Optional, Union

import torch
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam, _disable_pre_forward_hook
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map, optim_state_to_sharding_state


def _str_to_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    if name.startswith("torch."):
        name = name.replace("torch.", "", 1)
    if name.startswith("fp"):
        name = name.replace("fp", "float", 1)
    dtype = dict(
        float32=torch.float32,
        float=torch.float32,
        float64=torch.float64,
        double=torch.float64,
        float16=torch.float16,
        half=torch.float16,
        bfloat16=torch.bfloat16,
        bf16=torch.bfloat16,
        uint8=torch.uint8,
        byte=torch.uint8,
        int8=torch.int8,
        char=torch.int8,
        int16=torch.int16,
        short=torch.int16,
        int32=torch.int32,
        int=torch.int32,
        int64=torch.int64,
        long=torch.int64,
        bool=torch.bool,
    )[name]
    return dtype


class MegatronDistributedFusedAdam(DistributedFusedAdam):
    """Wrapper class that supports NeMo-Megatron optimizations

    When O2-style optimizations are enabled, gradients are accumulated
    into the main_grad buffer instead of the grad buffer.

    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[dict]],
        disable_distributed_parameters: bool = False,
        **kwargs,
    ):

        # Initialize process groups
        if 'process_group' not in kwargs and not parallel_state.is_unitialized():
            kwargs['process_group'] = parallel_state.get_data_parallel_group()
        if disable_distributed_parameters:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            self_groups = [torch.distributed.new_group(ranks=[i]) for i in range(world_size)]
            kwargs['distributed_process_group'] = self_groups[rank]
            kwargs['redundant_process_group'] = kwargs['process_group']

        # Make sure dtypes are in right type
        for keyword in ('dtype', 'grad_sync_dtype', 'param_sync_dtype'):
            if keyword in kwargs:
                kwargs[keyword] = _str_to_dtype(kwargs[keyword])

        # Make sure params are in consistent format (list of param group dicts)
        param_groups = list(params)
        assert param_groups
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        # Construct distributed optimizer
        super().__init__(param_groups, **kwargs)

        # Initialize weights that require FP32 grads
        if self.dtype != torch.float32 or self.grad_sync_dtype != torch.float32:
            fp32_params = []
            for param_group in param_groups:
                fp32_params.extend(
                    filter(lambda param: getattr(param, '_with_fp32_optimizer', False), param_group['params'],)
                )
            if fp32_params:
                assert self.dtype == torch.float32, (
                    'Param requires FP32 state, ' f'but optimizer is initialized with {dtype}'
                )
                self.init_params_bucket(
                    fp32_params, grad_sync_dtype=torch.float32,
                )

    def _make_post_backward_hook(self, param: torch.nn.Parameter, param_group_id: int, param_id: int,) -> Callable:
        def hook(*unused):
            if getattr(param, '_pre_forward_hook_is_enabled', False):
                raise RuntimeError(
                    'A parameter called its post-backward hook '
                    'before its pre-forward hook. '
                    'Please manually interact with the parameter '
                    'before the forward pass (e.g. by calling data_ptr) '
                    'or run DistributedFusedAdam with overlap_param_sync=False.'
                )
            with self._lock:
                need_to_initialize = 'fragments' not in self.state[param]
                if need_to_initialize:
                    self._init_param_state(param, param_group_id, param_id)
                if self.greedy_grad_copy and not getattr(param, '_disable_greedy_grad_copy', False):
                    self._grad_copy(param)
                    if self.overlap_grad_sync and not getattr(param, '_disable_overlap_grad_sync', False):
                        self._try_start_bucket_grad_sync(
                            params=[param], ignore_last_bucket=need_to_initialize,
                        )

        return hook

    def try_grad_sync(self, params: Iterable[torch.nn.Parameter]) -> None:
        def is_grad_copy_enabled(param: torch.nn.Parameter) -> bool:
            return not getattr(param, '_disable_greedy_grad_copy', False) and not getattr(
                param, '_disable_overlap_grad_sync', False
            )

        params = list(filter(is_grad_copy_enabled, params))
        for p in params:
            self._grad_copy(p)
        self._try_start_bucket_grad_sync(params=params)

    def zero_grad(self, *args, **kwargs) -> None:
        super().zero_grad(*args, **kwargs)

        # Reset main grads
        if self.contiguous_grad_buffer:
            for param in self.parameters():
                with _disable_pre_forward_hook(param):
                    param.main_grad = self.grad_buffer_view(param)

    def grad_norm(
        self, parameters: Optional[Iterable[torch.nn.Parameter]] = None, norm_type: float = 2.0, force: bool = False,
    ) -> torch.Tensor:
        assert norm_type == 2

        if parameters is not None:
            # Make sure we can access iterable multiple times
            parameters = list(parameters)

        # Compute grad norm
        if force or self._grad_norm is None:

            # Compute norm of local gradients for distributed optimizer
            grad_norm_sq = self._local_grad_norm(parameters=parameters, norm_type=norm_type,)
            if self.redundant_size > 1:
                grad_norm_sq /= self.redundant_size

            # Sum over all procs to get grad norm
            torch.distributed.all_reduce(
                grad_norm_sq, op=torch.distributed.ReduceOp.SUM,
            )
            self._grad_norm = grad_norm_sq.sqrt()

        # Use cached grad norm
        return super().grad_norm()

    def sharded_state_dict(self, model_sharded_state_dict):
        optimizer_state_dict = self.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict=model_sharded_state_dict, optim_params_iter=self.parameters(),
        )
        # Convert state
        step = optimizer_state_dict['state'].pop('step')
        state_dict_format = optimizer_state_dict.pop('format', None)
        optim_state_to_sharding_state(optimizer_state_dict, id_to_sharded_param_map)
        optimizer_state_dict['state']['step'] = step
        if state_dict_format is not None:
            optimizer_state_dict['format'] = state_dict_format

        def rename_fp32_params(x):
            if isinstance(x, ShardedTensor) and x.key.startswith('optimizer.state.param'):
                x.key = x.key.replace('optimizer.state.param', 'optimizer.state.fp32_param')
            return x

        dict_list_map_inplace(rename_fp32_params, optimizer_state_dict)

        return optimizer_state_dict
