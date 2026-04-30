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

from typing import List
from nemo.agents.voice_agent.utils.tool_calling.base import StandardSchemaTool, register_schema_tools_to_llm

__all__ = [
    "StandardSchemaTool",
    "register_schema_tools_to_llm",
    "ALL_STANDARD_SCHEMA_TOOLS",
    "register_standard_schema_tool",
    "get_standard_schema_tool",
    "list_standard_schema_tools",
]

ALL_STANDARD_SCHEMA_TOOLS = {}


def register_standard_schema_tool(cls):
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
    ALL_STANDARD_SCHEMA_TOOLS[key] = cls
    return cls


def get_standard_schema_tool(name: str, **kwargs) -> StandardSchemaTool:
    """
    Get a schema tool for evaluation by name.
    """
    if name not in ALL_STANDARD_SCHEMA_TOOLS:
        return None
    return ALL_STANDARD_SCHEMA_TOOLS[name](**kwargs)


def list_standard_schema_tools() -> List[StandardSchemaTool]:
    """
    List all schema tools for evaluation.
    """
    return list(ALL_STANDARD_SCHEMA_TOOLS.keys())
