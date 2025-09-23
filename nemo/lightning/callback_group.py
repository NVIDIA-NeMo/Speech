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

import atexit
import functools
from typing import Any, Callable, List, Optional

from lightning.pytorch.callbacks import Callback as PTLCallback
from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.one_logger_callback import OneLoggerNeMoCallback


class CallbackGroup:
    """A singleton registry to host and fan-out lifecycle callbacks.

    Other code should call methods on this group (e.g., `on_model_init_start`).
    The group will iterate all registered callbacks and, if a callback implements
    the method, invoke it with the provided arguments.
    """

    _instance: Optional['CallbackGroup'] = None

    @classmethod
    def get_instance(cls) -> 'CallbackGroup':
        """Get the singleton instance of CallbackGroup.

        Returns:
            CallbackGroup: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = CallbackGroup()
        return cls._instance

    def __init__(self) -> None:
        self._callbacks: List[BaseCallback] = [OneLoggerNeMoCallback()]
        # Ensure application-end is emitted at most once per process
        self._app_end_emitted: bool = False

    def register(self, callback: BaseCallback) -> None:
        """Register a callback to the callback group.

        Args:
            callback: The callback to register.
        """
        self._callbacks.append(callback)

    def update_config(self, nemo_version: str, trainer: Any, **kwargs) -> None:
        """Update configuration across all registered callbacks and attach them to trainer.

        Args:
            nemo_version: Version key (e.g., 'v1' or 'v2') for downstream config builders.
            trainer: Lightning Trainer to which callbacks should be attached if missing.
            **kwargs: Forwarded to each callback's update_config implementation.
        """
        # Forward update to each callback that supports update_config
        sanitized_group_callbacks: List[BaseCallback] = []
        for cb in self._callbacks:
            # Will ignore other callbacks like unittest.mock.MagicMock
            if not isinstance(cb, BaseCallback):
                continue
            if hasattr(cb, 'update_config'):
                method = getattr(cb, 'update_config')
                if callable(method):
                    method(nemo_version=nemo_version, trainer=trainer, **kwargs)
            sanitized_group_callbacks.append(cb)

        # Filter trainer callbacks to avoid leaking MagicMocks from tests
        existing = list(getattr(trainer, 'callbacks', []))
        sanitized_trainer_callbacks = [cb for cb in existing if isinstance(cb, PTLCallback)]

        callbacks = sanitized_group_callbacks + sanitized_trainer_callbacks

        # Sanitize callback state_key for pickling safety
        for cb in callbacks:
            try:
                key = getattr(cb, 'state_key', None)
                if not isinstance(key, str):
                    safe_key = (
                        f"{cb.__class__.__module__}.{getattr(cb.__class__, '__qualname__', cb.__class__.__name__)}"
                    )
                    setattr(cb, 'state_key', safe_key)
            except Exception:
                pass

        trainer.callbacks = callbacks

    @property
    def callbacks(self) -> List['BaseCallback']:
        """Get the list of registered callbacks.

        Returns:
            List[BaseCallback]: List of registered callbacks.
        """
        return self._callbacks

    def __getattr__(self, method_name: str) -> Callable:
        """Dynamically create a dispatcher for unknown attributes.

        Any attribute access is treated as a lifecycle method name.
        When invoked, the dispatcher will call that method on each registered
        callback if it exists.
        """

        def dispatcher(*args, **kwargs):
            for cb in self._callbacks:
                if hasattr(cb, method_name):
                    method = getattr(cb, method_name)
                    if callable(method):
                        method(*args, **kwargs)

        return dispatcher

    # Explicit idempotent app-end to avoid duplicate emissions across multiple callers
    def on_app_end(self, *args, **kwargs) -> None:
        """Emit application-end callbacks exactly once per process.

        Invokes `on_app_end` on each registered callback, if present. Subsequent
        calls are no-ops. All positional and keyword arguments are forwarded.
        """
        if self._app_end_emitted:
            return
        self._app_end_emitted = True
        for cb in self._callbacks:
            if hasattr(cb, 'on_app_end'):
                method = getattr(cb, 'on_app_end')
                if callable(method):
                    method(*args, **kwargs)


def hook_class_init_with_callbacks(cls, start_callback: str, end_callback: str) -> None:
    """Hook a class's __init__ to emit CallbackGroup start/end hooks.

    Args:
        cls (type): Class whose __init__ should be wrapped.
        start_callback (str): CallbackGroup method to call before __init__.
        end_callback (str): CallbackGroup method to call after __init__.
    """
    if not hasattr(cls, '__init__'):
        return

    original_init = cls.__init__

    # Idempotence guard: avoid wrapping the same __init__ multiple times (e.g., in multiple inheritance)
    if getattr(original_init, '_init_wrapped_for_callbacks', False):
        return

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        # Reentrancy guard: avoid double-emitting hooks across super().__init__ chains
        if getattr(self, '_in_wrapped_init', False):
            # If we're already inside a wrapped __init__, just call the original
            return original_init(self, *args, **kwargs)

        setattr(self, '_in_wrapped_init', True)
        group = CallbackGroup.get_instance()
        if hasattr(group, start_callback):
            getattr(group, start_callback)()
        result = original_init(self, *args, **kwargs)
        if hasattr(group, end_callback):
            getattr(group, end_callback)()
        return result

    wrapped_init._init_wrapped_for_callbacks = True
    cls.__init__ = wrapped_init


# Eagerly create the singleton on import so that early callers can use it
CallbackGroup.get_instance()

# Ensure that a single app-end is emitted at process shutdown (e.g., pytest end-of-session,
# non-Hydra entrypoints). Safe due to idempotent on_app_end.
atexit.register(lambda: CallbackGroup.get_instance().on_app_end())

__all__ = ['CallbackGroup', 'hook_class_init_with_callbacks']
