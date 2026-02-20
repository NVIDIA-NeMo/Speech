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

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from nemo.agents.voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo.agents.voice_agent.utils.tool_calling import StandardSchemaTool


@register_schema_tool_for_eval
class GetCityWeatherTool(StandardSchemaTool):
    """
    Get the weather of a city.
    """

    DESCRIPTION: str = """
        Get the weather of a city. You need to provide the city name to get the weather.
        """

    def __init__(self, *, description: Optional[str] = None):
        if description is None:
            description = self.DESCRIPTION
        super().__init__(description=description)

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.
        """
        return {
            "city_name": {
                "type": "string",
                "description": "The name of the city to get the weather of.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.
        """
        return ["city_name"]

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Get the weather of a city.
        """
        city_name = params.arguments.get("city_name")
        logger.debug(f"Getting weather of {city_name}")
        results = {
            "city": city_name,
            "weather": "sunny",
            "temperature": "20 degrees Celsius",
            "uv_index": "low",
        }
        await params.result_callback(results)


@register_schema_tool_for_eval
class ReadFileTool(StandardSchemaTool):
    """
    Read a file.
    """

    DESCRIPTION: str = """
        Read a file from the file system. You need to provide the file path to read the file.
        """

    def __init__(self):
        super().__init__(description=self.DESCRIPTION)

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.
        """
        return {
            "file_path": {
                "type": "string",
                "description": "The path of the file to read.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.
        """
        return ["file_path"]

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Read a file from the file system.
        """
        file_path = params.arguments.get("file_path")
        logger.debug(f"Reading file from {file_path}")
        try:
            with Path(file_path).open("r") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            await params.result_callback({"error": str(e)})
            return

        logger.debug(f"Loaded file {file_path} with content: `{content}`")
        results = {
            "file_path": file_path,
            "content": content,
        }
        await params.result_callback(results)
