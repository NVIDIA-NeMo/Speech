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

from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from nemo.agents.voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo.agents.voice_agent.utils.tool_calling import StandardSchemaTool


@register_schema_tool_for_eval
class GetMenuTool(StandardSchemaTool):
    """Returns the restaurant menu. Menu content is configured per scenario."""

    def __init__(self, *, menu: str = "", description: Optional[str] = None):
        super().__init__(description=description or "Get the restaurant menu to see available items and prices.")
        self.menu = menu

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    async def _execute(self, params: FunctionCallParams) -> None:
        logger.debug(f"GetMenuTool returning menu ({len(self.menu)} chars)")
        await params.result_callback({"menu": self.menu})
