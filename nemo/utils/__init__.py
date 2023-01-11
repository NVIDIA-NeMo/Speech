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


from nemo.utils.app_state import AppState
from nemo.utils.cast_utils import (
    CastToFloat,
    CastToFloatAll,
    avoid_bfloat16_autocast_context,
    avoid_float16_autocast_context,
    cast_all,
    cast_tensor,
)
from nemo.utils.nemo_logging import Logger as _Logger
from nemo.utils.nemo_logging import LogMode as logging_mode

logging = _Logger()
try:
    from nemo.utils.lightning_logger_patch import add_memory_handlers_to_pl_logger

    add_memory_handlers_to_pl_logger()
except ModuleNotFoundError:
    pass
