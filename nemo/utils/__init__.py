# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
    monkeypatched,
)
from nemo.utils.dtype import str_to_dtype
from nemo.utils.nemo_logging import Logger as _Logger
from nemo.utils.nemo_logging import LogMode as logging_mode

logging = _Logger()
try:
    from nemo.utils.lightning_logger_patch import add_memory_handlers_to_pl_logger

    add_memory_handlers_to_pl_logger()
except ModuleNotFoundError:
    pass

try:
    import webdataset
    from nemo.utils.data_utils import wds_url_opener

    webdataset.tariterators.url_opener = wds_url_opener
except ModuleNotFoundError:
    pass


def _register_omegaconf_safe_globals() -> None:
    """Allowlist omegaconf classes for torch's ``weights_only=True`` checkpoint loading.

    NeMo checkpoints store their configuration as omegaconf objects inside ``hyper_parameters``.
    torch >= 2.6 and lightning >= 2.6 load checkpoints with ``weights_only=True`` by default, which
    restricts unpickling to an allowlist of classes. The omegaconf container/node classes (and the
    builtin types their metadata references) are pure data holders that cannot trigger code
    execution on unpickling, so registering them keeps NeMo checkpoints loadable with the safe
    weights-only unpickler instead of forcing users back to ``weights_only=False``.
    """
    try:
        import collections

        import omegaconf
        import torch.serialization
    except ModuleNotFoundError:
        return
    if not hasattr(torch.serialization, "add_safe_globals"):
        return
    from typing import Any

    safe_classes = [
        omegaconf.DictConfig,
        omegaconf.ListConfig,
        omegaconf.base.ContainerMetadata,
        omegaconf.base.Metadata,
        Any,
        collections.defaultdict,
        dict,
        list,
        tuple,
        int,
        float,
        str,
        bool,
        bytes,
    ]
    # Typed value nodes; guard with getattr for omegaconf version differences.
    for node_name in (
        'AnyNode',
        'ValueNode',
        'StringNode',
        'IntegerNode',
        'FloatNode',
        'BooleanNode',
        'BytesNode',
        'PathNode',
        'EnumNode',
        'InterpolationResultNode',
    ):
        node_cls = getattr(omegaconf.nodes, node_name, None)
        if node_cls is not None:
            safe_classes.append(node_cls)
    torch.serialization.add_safe_globals(safe_classes)


_register_omegaconf_safe_globals()
