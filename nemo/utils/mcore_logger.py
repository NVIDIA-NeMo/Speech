# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging as _logging

from nemo.utils import logging as nemo_logger


def add_handlers_to_mcore_logger():
    """Add NeMo handlers to MCore loggers.

    MCore doesn't have and handlers for loggers (see
    https://docs.python.org/3/howto/logging-cookbook.html#adding-handlers-other-than-nullhandler-to-a-logger-in-a-library
    for a rationale). We have to add handlers explicitly.
    """
    mcore_logger = _logging.getLogger('megatron.core')
    for handler in nemo_logger._handlers.values():
        mcore_logger.addHandler(handler)
    mcore_logger.propagate = False
    mcore_logger.setLevel(nemo_logger._logger.level)
