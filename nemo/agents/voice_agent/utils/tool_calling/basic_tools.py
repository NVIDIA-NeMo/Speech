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

import asyncio
import python_weather
from loguru import logger
from pipecat.frames.frames import LLMTextFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams

HTTP_REQUEST_TIMEOUT = 10.0


async def tool_get_city_weather(params: FunctionCallParams, city_name: str):
    """Get the current weather of a city. The result includes city name, weather description,
    temperature, wind speed, wind direction, precipitation, humidity, visibility, and UV index.

    Args:
        city_name: The name of the city to get the weather of. For example, "London", "Beijing", "Paris".
                Other examples are: "Paris, TX, US", "Paris, FR" and "Tokyo, JP".
    """
    message = f"Looking up weather data for {city_name}. Please wait a moment..."
    # Send the message to upstream so that RTVI can log it while doesn't block the actual tool call.
    await params.llm.push_frame(LLMTextFrame(message), direction=FrameDirection.UPSTREAM)
    # Send the message to TTS directly so that the user can hear it immediately.
    await params.llm.push_frame(TTSSpeakFrame(message))

    # The measuring unit defaults to metric (Celsius)
    # Use imperial for Fahrenheit: python_weather.IMPERIAL
    async with python_weather.Client(unit=python_weather.METRIC) as client:
        # Fetch a weather forecast from a city
        logger.debug(f"Fetching weather forecast for `{city_name}`")
        try:
            weather: python_weather.Forecast = await asyncio.wait_for(
                client.get(city_name),
                timeout=HTTP_REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            error_msg = f"python_weather API request timed out after {HTTP_REQUEST_TIMEOUT} seconds for `{city_name}`"
            logger.error(error_msg)
            await params.result_callback({"error": error_msg})
            return
        except Exception as e:
            error_msg = f"Error fetching weather forecast for `{city_name}`: {str(e)}"
            logger.error(error_msg)
            await params.result_callback({"error": error_msg})
            return

        results = {
            "city": city_name,
            "description": str(weather.description),
            "temperature": f"{weather.temperature} degrees Celsius",
            "wind_speed": f"{weather.wind_speed} kilometers per hour",
            "wind_direction": str(weather.wind_direction.name),
            "precipitation": f"{weather.precipitation} millimeters",
            "humidity": f"{weather.humidity} percent",
            "visibility": f"{weather.visibility} kilometers",
            "uv_index": str(weather.ultraviolet),
        }
        logger.debug(f"Weather results for {city_name}: {results}")
        await params.result_callback(results)
