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

from typing import Callable, Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback

from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils import logging


def collect_precision(tensor: torch.Tensor) -> Dict[str, str]:
    """Returns tensor's precision"""
    if isinstance(tensor, torch.Tensor):
        return {"Precision": str(tensor.dtype)}
    else:
        return {"Precision": "not-a-tensor"}


def collect_precision_and_shape(tensor: torch.Tensor) -> Dict[str, str]:
    """Returns tensor's shape & precision"""
    if isinstance(tensor, torch.Tensor):
        return {"Shape": str(tensor.shape), "Precision": str(tensor.dtype)}
    else:
        return {"Shape": "not-a-tensor", "Precision": "not-a-tensor"}


class ParameterDebugger(Callback):
    """
    Debugging tool to help inspect parameters and gradients at any callback event.

    This callback handles the boilerplate needed to iterate over the model parameters and gradients,
    and applies user specified functions to them. These functions can be used to log attributes or
    apply asserts on the param and grad tensors. Attributes are logged in a table, with a row for each parameter name.
    Default behavior is to log the precision and shapes of each parameter and its gradient.

    Args:
        param_fn: Function to apply to model parameters. Can be used to apply assertions on the tensor,
            or return a mapping of labels and values to log for each parameter.
        grad_fn: Function to apply to model gradients. Can be used to apply assertions on the tensor,
            or return a mapping of labels and values to log for each gradient.
        log_on_hooks: PTL callback hook name or list of hook names on which to apply param_fn and grad_fn.
            See `PTL docs <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#hooks>`_ for more info
            on callback hooks. Note that some hooks that occur before the model is constructed are invalid.

    Example:
        >>> fn = lambda x: {"Norm": str(x.norm(2).item())}
        >>> callback = ParameterDebugger(param_fn=fn, log_on_hooks=["on_train_start", "on_train_end"])
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        param_fn: Optional[Callable[[torch.Tensor], Optional[Dict[str, str]]]] = collect_precision_and_shape,
        grad_fn: Optional[Callable[[torch.Tensor], Optional[Dict[str, str]]]] = collect_precision,
        log_on_hooks: Union[List[str], str] = "on_train_start",
    ):
        self.param_fn = param_fn
        self.grad_fn = grad_fn

        valid_hooks = set(
            [
                "teardown",
                "on_fit_end",
                "on_sanity_check_start",
                "on_sanity_check_end",
                "on_train_batch_start",
                "on_train_batch_end",
                "on_train_epoch_start",
                "on_train_epoch_end",
                "on_validation_epoch_start",
                "on_validation_epoch_end",
                "on_test_epoch_start",
                "on_test_epoch_end",
                "on_predict_epoch_start",
                "on_predict_epoch_end",
                "on_validation_batch_start",
                "on_validation_batch_end",
                "on_test_batch_start",
                "on_test_batch_end",
                "on_predict_batch_start",
                "on_predict_batch_end",
                "on_train_start",
                "on_train_end",
                "on_validation_start",
                "on_validation_end",
                "on_test_start",
                "on_test_end",
                "on_predict_start",
                "on_predict_end",
                "on_exception",
                "on_save_checkpoint",
                "on_load_checkpoint",
                "on_before_backward",
                "on_after_backward",
                "on_before_optimizer_step",
                "on_before_zero_grad",
            ]
        )

        if isinstance(log_on_hooks, str):
            log_on_hooks = [log_on_hooks]
        for hook_name in log_on_hooks:
            assert hook_name in valid_hooks, (
                "Hook {} supplied to log_on_hooks is not valid or " "can not be used. Valid hooks are {}"
            ).format(hook_name, valid_hooks)
            setattr(self, hook_name, self._apply_user_funcs)

    def _apply_user_funcs(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:
        """
        Iterate over model parameters, find gradient tensor, apply and collect outputs of
        param_fn and grad_fn, and log outputs in a table.
        """

        def find_grad_tensor(param: torch.Tensor) -> Optional[torch.Tensor]:
            """If using MCore optimizer, search the grad buckets for param's grad tensor."""
            if not isinstance(getattr(pl_module, 'optim', None), MegatronOptimizerModule):
                return param.grad

            for buf in pl_module.buffers:
                if param in buf.param_to_bucket:
                    return buf.param_to_bucket[param].grad_data

            return None

        names_col, params_output, grads_output = [], [], []
        for param_name, param_tensor in pl_module.named_parameters():
            grad_tensor = find_grad_tensor(param_tensor)
            short_name = param_name.replace("module.", "").replace(".weight", "")
            names_col.append(short_name)

            for tensor, fn, out_col in zip(
                [param_tensor, grad_tensor], [self.param_fn, self.grad_fn], [params_output, grads_output]
            ):
                if fn is not None:
                    if tensor is not None:
                        out_col.append(fn(tensor))
                    else:
                        out_col.append({})

        # get table column headers
        param_keys, grad_keys = set([]), set([])
        for output in params_output:
            if output is not None:
                param_keys.update(output.keys())
        for output in grads_output:
            if output is not None:
                grad_keys.update(output.keys())

        # create table only if there is something to print
        if any(param_keys) or any(grad_keys):
            from prettytable import PrettyTable

            debug_table = PrettyTable()
            debug_table.add_column("Parameter", names_col)

            for prefix, keys, output_list in zip(
                ["Param ", "Grad "], [param_keys, grad_keys], [params_output, grads_output]
            ):
                for k in keys:
                    col_to_log = []
                    for output in output_list:
                        if output is not None:
                            col_to_log.append(output.get(k, None))
                        else:
                            col_to_log.append(None)
                    if col_to_log != []:
                        debug_table.add_column(prefix + k, col_to_log)

            debug_table.align = "l"
            logging.info("\n" + debug_table.get_string())
