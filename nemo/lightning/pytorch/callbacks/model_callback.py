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


import inspect
from typing import Callable, Optional
from lightning.pytorch.callbacks import LambdaCallback


class ModelCallback(LambdaCallback):
    """
    A callback that extends LambdaCallback to intelligently handle function parameters.
    Functions can take either (trainer, pl_module), just (pl_module), or just (trainer).

    Supported parameter names:
    - trainer, pl_trainer
    - model, pl_model, pl_module, module

    Example:
        >>> # Using with torch.compile
        >>> callback = ModelCallback(on_train_start=torch.compile)
        >>>
        >>> # Using with thunder_compile
        >>> callback = ModelCallback(on_train_start=thunder_compile)
        >>>
        >>> # Mix different callbacks
        >>> callback = ModelCallback(
        ...     on_train_start=lambda model: torch.compile(model),
        ...     on_fit_start=lambda trainer, model: print(f"Starting fit with {model}")
        ... )
    """

    TRAINER_PARAMS = {'trainer', 'pl_trainer'}
    MODEL_PARAMS = {'model', 'pl_model', 'pl_module', 'module'}

    def __init__(
        self,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        on_fit_start: Optional[Callable] = None,
        on_fit_end: Optional[Callable] = None,
        on_sanity_check_start: Optional[Callable] = None,
        on_sanity_check_end: Optional[Callable] = None,
        on_train_batch_start: Optional[Callable] = None,
        on_train_batch_end: Optional[Callable] = None,
        on_train_epoch_start: Optional[Callable] = None,
        on_train_epoch_end: Optional[Callable] = None,
        on_validation_epoch_start: Optional[Callable] = None,
        on_validation_epoch_end: Optional[Callable] = None,
        on_test_epoch_start: Optional[Callable] = None,
        on_test_epoch_end: Optional[Callable] = None,
        on_validation_batch_start: Optional[Callable] = None,
        on_validation_batch_end: Optional[Callable] = None,
        on_test_batch_start: Optional[Callable] = None,
        on_test_batch_end: Optional[Callable] = None,
        on_train_start: Optional[Callable] = None,
        on_train_end: Optional[Callable] = None,
        on_validation_start: Optional[Callable] = None,
        on_validation_end: Optional[Callable] = None,
        on_test_start: Optional[Callable] = None,
        on_test_end: Optional[Callable] = None,
        on_exception: Optional[Callable] = None,
        on_save_checkpoint: Optional[Callable] = None,
        on_load_checkpoint: Optional[Callable] = None,
        on_before_backward: Optional[Callable] = None,
        on_after_backward: Optional[Callable] = None,
        on_before_optimizer_step: Optional[Callable] = None,
        on_before_zero_grad: Optional[Callable] = None,
        on_predict_start: Optional[Callable] = None,
        on_predict_end: Optional[Callable] = None,
        on_predict_batch_start: Optional[Callable] = None,
        on_predict_batch_end: Optional[Callable] = None,
        on_predict_epoch_start: Optional[Callable] = None,
        on_predict_epoch_end: Optional[Callable] = None,
    ):
        # Create a dictionary of non-None callbacks
        callbacks = {
            name: self._wrap_func(func)
            for name, func in locals().items()
            if name != 'self' and name != '__class__' and func is not None
        }

        super().__init__(**callbacks)

    def _get_param_type(self, param_name: str) -> Optional[str]:
        """Determine if a parameter name refers to trainer or model."""
        param_name = param_name.lower()
        if param_name in self.TRAINER_PARAMS:
            return 'trainer'
        if param_name in self.MODEL_PARAMS:
            return 'model'
        return None

    def _wrap_func(self, func: Callable) -> Callable:
        """Wraps a function to handle parameter inspection and passing."""
        sig = inspect.signature(func)
        params = sig.parameters

        def wrapped(trainer, pl_module, *args, **kwargs):
            call_args = {}

            for param_name, param in params.items():
                param_type = self._get_param_type(param_name)

                if param_type == 'trainer':
                    call_args[param_name] = trainer
                elif param_type == 'model':
                    call_args[param_name] = pl_module
                else:
                    # If parameter name is not recognized, use position to determine
                    if len(params) == 1:
                        call_args[param_name] = pl_module
                    elif len(params) == 2:
                        if len(call_args) == 0:
                            call_args[param_name] = trainer
                        else:
                            call_args[param_name] = pl_module
                    else:
                        raise ValueError(
                            f"Unable to determine parameter mapping for '{param_name}'. "
                            f"Please use recognized parameter names: "
                            f"trainer/pl_trainer for trainer, "
                            f"model/pl_model/pl_module/module for model."
                        )

            try:
                return func(**call_args)
            except TypeError as e:
                raise TypeError(
                    f"Failed to call callback function {func.__name__ if hasattr(func, '__name__') else func}. "
                    f"Attempted to pass arguments: {call_args.keys()}. Error: {str(e)}"
                ) from e

        return wrapped
