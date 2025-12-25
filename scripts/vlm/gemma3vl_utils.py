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
"""Utils for training recipes."""

import signal
import warnings
from nemo.utils import logging


def filter_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="typing_extensions")
    warnings.filterwarnings("ignore", module="megatron.core.distributed.param_and_grad_buffer")
    warnings.filterwarnings("ignore", message=r".*deprecated.*")


def filter_grad_bucket_logs():
    """Filter the noisy `Number of buckets...` log dumped by megatron."""

    def _filter(record):
        del record
        return False

    for handler in logging._logger.handlers:
        handler.addFilter(_filter)


def ignore_sigprof():
    signal.signal(signal.SIGPROF, signal.SIG_IGN)
