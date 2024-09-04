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

import logging
from typing import Any, Callable, Generic, TypeVar, Union, overload

T = TypeVar("T", bound=Callable[..., Any])

try:
    import nemo_run as run

    Config = run.Config
    Partial = run.Partial
except ImportError:
    logging.warning(
        "Trying to use Config or Partial, but NeMo-Run is not installed. Please install NeMo-Run before proceeding."
    )

    _T = TypeVar("_T")

    class Config(Generic[_T]):
        pass

    class Partial(Generic[_T]):
        pass


def task(*args: Any, **kwargs: Any) -> Callable[[T], T]:
    try:
        import nemo_run as run

        return run.task(*args, **kwargs)
    except (ImportError, AttributeError):
        # Return a no-op function
        def noop_decorator(func: T) -> T:
            return func

        return noop_decorator


@overload
def factory() -> Callable[[T], T]: ...


@overload
def factory(*args: Any, **kwargs: Any) -> Callable[[T], T]: ...


def factory(*args: Any, **kwargs: Any) -> Union[Callable[[T], T], T]:
    try:
        import nemo_run as run

        if not args:
            return run.factory(**kwargs)
        else:
            # Used as @factory(*args, **kwargs)
            return run.factory(*args, **kwargs)
    except (ImportError, AttributeError):
        # Return a no-op function
        def noop_decorator(func: T) -> T:
            return func

        if not args and not kwargs:
            return noop_decorator
        else:
            return noop_decorator
