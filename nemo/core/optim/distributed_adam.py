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

import torch
from apex.contrib.optimizers.distributed_fused_adam import (
    DistributedFusedAdam,
    _coalescing_manager,
    _coalescing_manager_append_work,
    _disable_pre_forward_hook,
)
from megatron.core import parallel_state


def _str_to_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    if name in ('', 'none'):
        return torch.float32
    elif name in ('torch.float32', 'float32', 'float', 'fp32', '32'):
        return torch.float32
    elif name in ('torch.float16', 'float16', 'half', 'fp16', '16'):
        return torch.float16
    elif name in ('torch.bfloat16', 'bfloat16', 'bf16'):
        return torch.bfloat16
    else:
        raise ValueError(f'unsupported dtype ({dtype})')


class MegatronDistributedFusedAdam(DistributedFusedAdam):
    """Wrapper class that supports NeMo-Megatron optimizations

    When O2-style optimizations are enabled, gradients are accumulated
    into the main_grad buffer instead of the grad buffer.

    """

    def __init__(self, params, disable_distributed_parameters=False, **kwargs):

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

        # Check if explicit FP32 optimizer is needed
        self._fp32_optim = None
        distopt_param_groups = param_groups
        dtype = kwargs['dtype'] if 'dtype' in kwargs else torch.float32
        grad_sync_dtype = kwargs['grad_sync_dtype'] if 'grad_sync_dtype' in kwargs else dtype
        needs_fp32_optimizer = any(
            getattr(param, '_with_fp32_optimizer', False)
            for param in itertools.chain.from_iterable(param_group['params'] for param_group in param_groups)
        )
        if (dtype != torch.float32 or grad_sync_dtype != torch.float32) and needs_fp32_optimizer:

            # Find params that require explicit FP32 optimizer
            distopt_param_groups = []
            fp32_param_groups = []
            self._fp32_optim_main_params = collections.OrderedDict()
            for param_group in param_groups:
                distopt_param_group = {key: val for key, val in param_group.items() if key != 'params'}
                distopt_param_group['params'] = []
                fp32_param_group = {key: val for key, val in param_group.items() if key != 'params'}
                fp32_param_group['params'] = []
                for model_param in param_group['params']:
                    if getattr(model_param, '_with_fp32_optimizer', False):
                        main_param = model_param.detach().clone().float()
                        fp32_param_group['params'].append(main_param)
                        self._fp32_optim_main_params[model_param] = main_param
                    else:
                        distopt_param_group['params'].append(model_param)
                distopt_param_groups.append(distopt_param_group)
                fp32_param_groups.append(fp32_param_group)

            # Construct explicit FP32 optimizer
            adamw_kwargs = {}
            for name in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad'):
                if name in kwargs:
                    adamw_kwargs[name] = kwargs[name]
            self._fp32_optim = torch.optim.AdamW(fp32_param_groups, **adamw_kwargs)
            self._fp32_optim_grad_sync_needed = True

        # Construct distributed optimizer
        super().__init__(distopt_param_groups, **kwargs)

    def _make_post_backward_hook(self, param, param_group_id, param_id):
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

    def _filter_distopt_params(self, params):
        if self._fp32_optim is None:
            return params
        if params is None:
            return None
        if isinstance(params, torch.Tensor):
            params = [params]
        return filter(lambda param: param not in self._fp32_optim_main_params, params)

    def parameters(self, with_fp32_optim_params=False):
        if with_fp32_optim_params and self._fp32_optim is not None:
            return itertools.chain(super().parameters(), self._fp32_optim_main_params.keys())
        else:
            return super().parameters()

    def init_params(self, params=None):
        super().init_params(self._filter_distopt_params(params))

    def init_params_bucket(self, params):
        super().init_params_bucket(self._filter_distopt_params(params))

    def try_grad_sync(self, params):
        params = self._filter_distopt_params(params)
        params = [p for p in params if not getattr(p, '_disable_greedy_grad_copy', False)]
        params = [p for p in params if not getattr(p, '_disable_overlap_grad_sync', False)]
        for p in params:
            self._grad_copy(p)
        self._try_start_bucket_grad_sync(params=params)

    def _try_start_bucket_param_sync(self, params=None):
        super()._try_start_bucket_param_sync(self._filter_distopt_params(params))

    def _fp32_optim_grad_sync(self):
        if self._fp32_optim is None or not self._fp32_optim_grad_sync_needed:
            return
        for model_param, main_param in self._fp32_optim_main_params.items():
            if model_param.grad is not None:
                main_param.grad += model_param.grad.detach()
        with _coalescing_manager(self.process_group, self.device, async_ops=True) as cm:
            for main_param in self._fp32_optim_main_params.values():
                _coalescing_manager_append_work(
                    cm,
                    torch.distributed.all_reduce(
                        main_param.grad, op=torch.distributed.ReduceOp.AVG, group=self.process_group, async_op=True,
                    ),
                )
        cm.wait()
        self._fp32_optim_grad_sync_needed = False

    def zero_grad(self, *args, **kwargs):
        super().zero_grad(*args, **kwargs)

        # Reset grads for explicit FP32 optimizer
        if self._fp32_optim is not None:
            self._fp32_optim_grad_sync_needed = True
            self._fp32_optim.zero_grad(set_to_none=False)
            for model_param, main_param in self._fp32_optim_main_params.items():
                if main_param.grad is None:
                    main_param.grad = torch.zeros_like(main_param)
                if model_param.grad is not None:
                    model_param.grad.zero_()
                model_param.main_grad = main_param.grad

        # Reset main grads
        if self.contiguous_grad_buffer:
            for param in self.parameters():
                with _disable_pre_forward_hook(param):
                    param.main_grad = self.grad_buffer_view(param)

    def grad_norm(self, parameters=None, norm_type=2.0, force=False):
        assert norm_type == 2

        if parameters is not None:
            # Make sure we can access iterable multiple times
            parameters = list(parameters)

        # Compute grad norm
        if force or self._grad_norm is None:

            # Compute norm of local gradients for distributed optimizer
            grad_norm_sq = self._local_grad_norm(
                parameters=self._filter_distopt_params(parameters), norm_type=norm_type,
            )
            if self.redundant_size > 1:
                grad_norm_sq /= self.redundant_size

            # Compute norm of local gradients for explicit FP32 optimizer
            if self._fp32_optim is not None:
                self._fp32_optim_grad_sync()
                if parameters is None:
                    for main_param in self._fp32_optim_main_params.values():
                        grad_norm_sq += torch.linalg.norm(main_param.grad) ** 2 / self.process_group_size
                else:
                    for model_param in parameters:
                        if model_param in self._fp32_optim_main_params:
                            main_param = self._fp32_optim_main_params[model_param]
                            grad_norm_sq += torch.linalg.norm(main_param.grad) ** 2 / self.process_group_size

            # Sum over all procs to get grad norm
            torch.distributed.all_reduce(
                grad_norm_sq, op=torch.distributed.ReduceOp.SUM,
            )
            self._grad_norm = grad_norm_sq.sqrt()

        # Use cached grad norm
        return super().grad_norm()

    def step(self, closure=None, *, grad_scaler=None):

        # Apply distributed optimizer
        loss = super().step(closure=closure, grad_scaler=grad_scaler)

        if self._fp32_optim is not None:

            # Handle grad scaling
            if grad_scaler is not None:
                scaler_state = grad_scaler._per_optimizer_states[id(self)]
                for _, found_inf in scaler_state['found_inf_per_device'].items():
                    if found_inf.item():
                        return loss

            # Update learning rate
            for distopt_group, fp32_optim_group in zip(self.param_groups, self._fp32_optim.param_groups):
                fp32_optim_group['lr'] = distopt_group['lr']

            # Apply explicit FP32 optimizer
            self._fp32_optim_grad_sync()
            for main_param in self._fp32_optim_main_params.values():
                main_param.grad *= self._grad_scale
            self._fp32_optim.step()
            for model_param, main_param in self._fp32_optim_main_params.items():
                model_param.detach().copy_(main_param.detach())

        return loss

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self._fp32_optim is not None and state_dict is not None:
            state_dict['fp32_optim'] = self._fp32_optim.state_dict()
            state_dict['fp32_optim_fp32_params'] = list(self._fp32_optim_main_params.values())
        return state_dict

    def load_state_dict(self, state_dict):
        if self._fp32_optim is not None and 'fp32_optim' in state_dict:
            self._fp32_optim.load_state_dict(state_dict['fp32_optim'])
            del state_dict['fp32_optim']
            for old_main_param, new_main_param in zip(
                self._fp32_optim_main_params.values(), state_dict['fp32_optim_fp32_params']
            ):
                old_main_param.copy_(new_main_param.detach())
            del state_dict['fp32_optim_fp32_params']
        return super().load_state_dict(state_dict)
