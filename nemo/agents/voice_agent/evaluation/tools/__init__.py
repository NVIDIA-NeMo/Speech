# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, List, Optional

from pipecat.processors.frameworks.rtvi import RTVIProcessor

from nemo.agents.voice_agent.utils.tool_calling.base import StandardSchemaTool

ALL_SCHEMA_TOOLS_FOR_EVAL: Dict[str, StandardSchemaTool] = {}


def register_schema_tool_for_eval(cls):
    """Class decorator that registers a tool class into ALL_STANDARD_SCHEMA_TOOLS.

    Usage:
        @register_standard_schema_tool
        class MyTool:
            name = "my_tool"
            ...

    The tool is keyed by cls.name if it exists, otherwise cls.__name__.
    """
    if not issubclass(cls, StandardSchemaTool):
        raise ValueError(f"Class {cls.__name__} is not a subclass of StandardSchemaTool")
    key = getattr(cls, "name", cls.__name__)
    ALL_SCHEMA_TOOLS_FOR_EVAL[key] = cls
    return cls


def get_schema_tool_for_eval(name: str, rtvi: Optional[RTVIProcessor] = None, **kwargs) -> StandardSchemaTool:
    """
    Get a schema tool for evaluation by name, and initialize the tool with the given arguments.

    Args:
        name: The name of the tool.
        rtvi: The RTVI processor to use for sending messages to the evaluator.
        kwargs: The additional keyword arguments to pass to the tool constructor.

    Returns:
        The schema tool for evaluation.
    """
    if name not in ALL_SCHEMA_TOOLS_FOR_EVAL:
        return None
    tool_class = ALL_SCHEMA_TOOLS_FOR_EVAL[name]
    # if `rtvi` is in class signature, pass it to the tool constructor
    if "rtvi" in inspect.signature(tool_class).parameters:
        return tool_class(rtvi=rtvi, **kwargs)
    return tool_class(**kwargs)


def list_schema_tools_for_eval() -> List[StandardSchemaTool]:
    """
    List all schema tools for evaluation.
    """
    return list(ALL_SCHEMA_TOOLS_FOR_EVAL.keys())


import nemo.agents.voice_agent.evaluation.tools.basic_tools

# Import subpackages to trigger @register_schema_tool_for_eval decorators.
# Must be at the end to avoid circular imports (data modules import register_schema_tool_for_eval).
import nemo.agents.voice_agent.evaluation.tools.rtvi_control
