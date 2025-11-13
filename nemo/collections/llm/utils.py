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

import logging
from typing import Any, Callable, Generic, TypeVar, Union, overload

import torch
import torch.distributed as dist

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
        """ """

        pass

    class Partial(Generic[_T]):
        """ """

        pass


SAFE_REPOS = [
    "nvidia",
    "Qwen",
    "deepseek-ai",
    "meta-llama",
    "google",
    "openai",
    "mistralai",
    "moonshotai",
    "llava-hf",
    "gpt2",
    "baichuan-inc",
]


def task(*args: Any, **kwargs: Any) -> Callable[[T], T]:
    """ """

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
    """ """

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


def torch_dtype_from_precision(precision: Union[int, str]) -> torch.dtype:
    """Mapping from PTL precision types to corresponding PyTorch parameter datatype."""

    if precision in ('bf16', 'bf16-mixed'):
        return torch.bfloat16
    elif precision in (16, '16', '16-mixed'):
        return torch.float16
    elif precision in (32, '32', '32-true'):
        return torch.float32
    else:
        raise ValueError(f"Could not parse the precision of `{precision}` to a valid torch.dtype")


def barrier():
    """Waits for all processes."""
    if dist.is_initialized():
        dist.barrier()


def is_safe_repo(trust_remote_code: bool = None, hf_path: str = None) -> bool:
    """
    Determine whether remote code execution should be trusted for a given
    Hugging Face repository path.
    Args:
        trust_remote_code (bool): whther to define repo as safe w/o checking SAFE_REPOS.
        hf_path (str): path to HF's model or dataset.
    Returns:
        True if remote code execution is allowed; False otherwise.
    """
    if trust_remote_code is not None:
        return trust_remote_code

    hf_repo = hf_path.split("/")[0]
    if hf_repo in SAFE_REPOS:
        return True

    return False
