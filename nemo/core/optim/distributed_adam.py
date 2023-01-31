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

import torch
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
from apex.transformer import parallel_state


# Wrapper class that supports main_grad buffer
# Note: main_grad buffer is used for O2-style optimizations
class MegatronDistributedFusedAdam(DistributedFusedAdam):
    def __init__(self, *args, disable_distributed_parameters=False, **kwargs):
        if 'process_group' not in kwargs and not parallel_state.is_unitialized():
            kwargs['process_group'] = parallel_state.get_data_parallel_group()
        if disable_distributed_parameters:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            self_groups = [torch.distributed.new_group(ranks=[i]) for i in range(world_size)]
            kwargs['distributed_process_group'] = self_groups[rank]
            kwargs['redundant_process_group'] = kwargs['process_group']
        super().__init__(*args, **kwargs)

    def _make_post_backward_hook(self, param, param_group_id, param_id):
        def hook(*unused):
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

    def try_grad_sync(self, params):
        params = list(params)
        for p in params:
            self._grad_copy(p)
        self._try_start_bucket_grad_sync(params=params)

    def zero_grad(self, *args, **kwargs):
        super().zero_grad(*args, **kwargs)
        if self.contiguous_grad_buffer:
            for param in self.parameters():
                param.main_grad = self.grad_buffer_view(param)
                if param.dtype == param.main_grad.dtype and param.is_cuda:
                    param.grad = param.main_grad
