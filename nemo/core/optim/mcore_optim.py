# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.utils.nvtx import nvtx_range_pop, nvtx_range_push


def _filter_empty_common_step(state_dict):
    """
    Filters out the 'common_step' key from the optimizer state dictionary if its value is None.
    This prevents errors during state loading when 'common_step' is unintentionally included.

    Args:
        state_dict (dict): The optimizer state dictionary.
    """
    try:
        common_step = state_dict['optimizer']['state']['common_step']

        if common_step is None:
            del state_dict['optimizer']['state']['common_step']
    except KeyError:
        pass


class McoreDistributedOptimizer(torch.optim.Optimizer):
    """
    A wrapper for the Megatron Core distributed optimizer.
    This class extends the base optimizer functionality and provides additional state
    management and checkpointing capabilities.

    Args:
        optim (MegatronOptimizer): The distributed optimizer from Megatron Core.
    """

    NVTX_LABEL = "nemo.core.optim.mcore_optim"

    def __init__(self, optim):
        self.defaults = {}
        self.mcore_optimizer = optim

    def zero_grad(self, set_to_none: bool = True):
        """
        We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point.

        Args:
            set_to_none (bool, optional): Whether to set gradients to None instead of zero.
                                          Defaults to True.
        """
        self.mcore_optimizer.zero_grad(set_to_none)

    def reload_model_params(self, state_dict=None):
        """
        Reloads model parameters from the optimizer.
        """
        if state_dict is None:
            self.mcore_optimizer.reload_model_params()
        else:
            self.mcore_optimizer.reload_model_params(state_dict=state_dict)

    def state_dict(self):
        """
        Returns the state dictionary of the optimizer.

        Returns:
            dict: The state dictionary containing optimizer states.
        """
        return self.mcore_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state from a given state dictionary.
        Also filters out unnecessary keys before loading.

        Args:
            state_dict (dict): The optimizer state dictionary.
        """
        _filter_empty_common_step(state_dict)
        self.mcore_optimizer.load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict, optimizer_state_dict=None, is_loading=False, dist_ckpt_parallel_save=False
    ):
        """
        Returns the sharded state dictionary for distributed checkpointing.

        Args:
            model_sharded_state_dict (dict): The model's sharded state dictionary.
            optimizer_state_dict (dict, optional): The optimizer's state dictionary. Defaults to None.
            is_loading (bool, optional): Whether the function is being used for loading. Defaults to False.
            dist_ckpt_parallel_save (bool, optional): Flag indicating whether to use a fully sharded model
                space. Defaults to False.

        Returns:
            dict: The sharded optimizer state dictionary.
        """
        sharding_type = 'fully_sharded_model_space' if dist_ckpt_parallel_save else 'dp_zero_gather_scatter'
        return self.mcore_optimizer.sharded_state_dict(
            model_sharded_state_dict, is_loading=is_loading, sharding_type=sharding_type
        )

    def step(self, closure=None):
        """
        Performs a single optimization step, including gradient clipping if needed.
        Always return successful since there is no overflow

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss. Defaults to None.

        Returns:
            tuple: Contains (loss, grad_norm, num_zeros_in_grad).
        """
        # Apply closure
        loss = None
        if closure is not None:
            nvtx_range_push(f"{McoreDistributedOptimizer.NVTX_LABEL}.step.closure")
            loss = closure()
            nvtx_range_pop(f"{McoreDistributedOptimizer.NVTX_LABEL}.step.closure")

        # return unused update_successful, grad_norm, num_zeros_in_grad
        nvtx_range_push(f"{McoreDistributedOptimizer.NVTX_LABEL}.step.step")
        _, grad_norm, num_zeros_in_grad = self.mcore_optimizer.step()
        nvtx_range_pop(f"{McoreDistributedOptimizer.NVTX_LABEL}.step.step")

        return loss, grad_norm, num_zeros_in_grad

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        """
        Retrieves the optimizer state.

        Returns:
            dict: The optimizer state dictionary.
        """
        return (
            self.mcore_optimizer.state
            if hasattr(self, 'mcore_optimizer') and hasattr(self.mcore_optimizer, 'state')
            else {}
        )

    def _set_state(self, value):
        """
        Sets the optimizer state.

        Args:
            value (dict): The new optimizer state.
        """
        self.mcore_optimizer.state = value

    state = property(_get_state, _set_state)

    def save_parameter_state(self, filename: str):
        """
        Saves the optimizer parameter state to a file.

        Args:
            filename (str): The file path to save the parameter state.
        """
        self.mcore_optimizer.save_parameter_state(filename)

    def load_parameter_state(self, filename: str):
        """
        Loads the optimizer parameter state from a file.

        Args:
            filename (str): The file path from which to load the parameter state.
        """
        self.mcore_optimizer.load_parameter_state(filename)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        """
        Retrieves the parameter groups of the optimizer.

        Returns:
            list: The parameter groups.
        """
        return self.mcore_optimizer.param_groups if hasattr(self, 'mcore_optimizer') else []

    def _set_param_groups(self, value):
        """
        Sets the parameter groups of the optimizer.

        Args:
            value (list): The new parameter groups.
        """
        self.mcore_optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    def disable_pre_hook(self):
        """
        Disables any pre-hooks applied to the optimizer.
        """
        self.mcore_optimizer.disable_pre_hook()

    def enable_pre_hook(self):
        """
        Enables pre-hooks for the optimizer.
        """
        self.mcore_optimizer.enable_pre_hook()
