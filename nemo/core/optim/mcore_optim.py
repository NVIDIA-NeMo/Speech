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

import torch

try:
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class McoreDistributedOptimizer(torch.optim.Optimizer):
    """
    A wrapper for Mcore distributed optimizer.

    Arguments:
        optim: distributed optimizer from Megatron core.
    """

    def __init__(self, optim):
        self.defaults = {}
        self.mcore_optimizer = optim
        self.param_groups = self.mcore_optimizer.param_groups
        self.state = self.mcore_optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        self.mcore_optimizer.zero_grad(set_to_none)

    def reload_model_params(self):
        self.mcore_optimizer.reload_model_params()

    def state_dict(self):
        return self.mcore_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.mcore_optimizer.load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict, optimizer_state_dict=None, is_loading=False, dist_ckpt_parallel_save=False
    ):
        sharding_type = 'fully_sharded_model_space' if dist_ckpt_parallel_save else 'dp_zero_gather_scatter'
        return self.mcore_optimizer.sharded_state_dict(
            model_sharded_state_dict, is_loading=is_loading, sharding_type=sharding_type
        )

    def step(self, closure):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""
        # Apply closure
        loss = None
        if closure is not None:
            loss = closure()

        # return unused update_successful, grad_norm, num_zeros_in_grad
        self.mcore_optimizer.step()

        return loss

    def save_parameter_state(self, filename: str):
        self.mcore_optimizer.save_parameter_state(filename)

    def load_parameter_state(self, filename: str):
        self.mcore_optimizer.load_parameter_state(filename)

    def finish_param_sync(self, model_index):
        self.mcore_optimizer.finish_param_sync(model_index)

    def disable_pre_hook(self):
        self.mcore_optimizer.disable_pre_hook()

    def enable_pre_hook(self):
        self.mcore_optimizer.enable_pre_hook()
