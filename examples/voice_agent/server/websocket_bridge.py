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
import json
from typing import Optional

import websockets
from loguru import logger
from websockets.client import WebSocketClientProtocol


class WebsocketBridge:
    """
    A bridge that connects two websocket endpoints and forwards messages between them.
    Useful for connecting two voice agents to have them communicate with each other.
    """

    def __init__(
        self,
        agent1_url: str,
        agent2_url: str,
        agent1_name: str = "Agent1",
        agent2_name: str = "Agent2",
        log_messages: bool = True,
        filter_audio: bool = False,
    ):
        """
        Initialize the websocket bridge.

        Args:
            agent1_url: WebSocket URL for the first agent
            agent2_url: WebSocket URL for the second agent
            agent1_name: Name for logging purposes
            agent2_name: Name for logging purposes
            log_messages: Whether to log message traffic
            filter_audio: Whether to filter audio frames from logging
        """
        self.agent1_url = agent1_url
        self.agent2_url = agent2_url
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.log_messages = log_messages
        self.filter_audio = filter_audio

        self.agent1_ws: Optional[WebSocketClientProtocol] = None
        self.agent2_ws: Optional[WebSocketClientProtocol] = None
        self.running = False
        self.tasks = []

    async def connect(self):
        """Connect to both websocket endpoints."""
        logger.info(f"Connecting to {self.agent1_name} at {self.agent1_url}")
        self.agent1_ws = await websockets.connect(self.agent1_url)
        logger.info(f"Connected to {self.agent1_name}")

        logger.info(f"Connecting to {self.agent2_name} at {self.agent2_url}")
        self.agent2_ws = await websockets.connect(self.agent2_url)
        logger.info(f"Connected to {self.agent2_name}")

    async def disconnect(self):
        """Disconnect from both websocket endpoints."""
        logger.info("Disconnecting from agents...")

        if self.agent1_ws:
            await self.agent1_ws.close()
            logger.info(f"Disconnected from {self.agent1_name}")

        if self.agent2_ws:
            await self.agent2_ws.close()
            logger.info(f"Disconnected from {self.agent2_name}")

    def _should_log_message(self, message) -> bool:
        """Determine if a message should be logged."""
        if not self.log_messages:
            return False

        if self.filter_audio:
            # Try to parse as JSON to check message type
            try:
                if isinstance(message, bytes):
                    # Binary data, likely audio - skip if filtering
                    return False
                elif isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type", "")
                    # Skip audio-related messages if filtering
                    if "audio" in msg_type.lower():
                        return False
            except (json.JSONDecodeError, AttributeError):
                pass

        return True

    def _format_message_preview(self, message, max_length: int = 100) -> str:
        """Format a message for logging with preview."""
        if isinstance(message, bytes):
            if len(message) > max_length:
                return f"<binary data, {len(message)} bytes>"
            return f"<binary data, {len(message)} bytes>: {message[:50].hex()}..."
        else:
            msg_str = str(message)
            if len(msg_str) > max_length:
                return msg_str[:max_length] + "..."
            return msg_str

    async def _forward_messages(
        self, source_ws: WebSocketClientProtocol, dest_ws: WebSocketClientProtocol, source_name: str, dest_name: str
    ):
        """
        Forward messages from source websocket to destination websocket.

        Args:
            source_ws: Source websocket to receive from
            dest_ws: Destination websocket to send to
            source_name: Name of source for logging
            dest_name: Name of destination for logging
        """
        try:
            async for message in source_ws:
                if self._should_log_message(message):
                    preview = self._format_message_preview(message)
                    logger.debug(f"{source_name} -> {dest_name}: {preview}")

                # Forward the message to the destination
                if isinstance(message, bytes):
                    await dest_ws.send(message)
                else:
                    await dest_ws.send(message)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed for {source_name}: {e}")
            self.running = False
        except Exception as e:
            logger.error(f"Error forwarding messages from {source_name} to {dest_name}: {e}")
            self.running = False

    async def run(self):
        """
        Run the bridge, forwarding messages bidirectionally between the two agents.
        This will run until either connection closes or an error occurs.
        """
        if not self.agent1_ws or not self.agent2_ws:
            raise RuntimeError("Must call connect() before run()")

        self.running = True
        logger.info(f"Starting bidirectional message forwarding between {self.agent1_name} and {self.agent2_name}")

        # Create tasks for bidirectional forwarding
        task1 = asyncio.create_task(
            self._forward_messages(self.agent1_ws, self.agent2_ws, self.agent1_name, self.agent2_name)
        )
        task2 = asyncio.create_task(
            self._forward_messages(self.agent2_ws, self.agent1_ws, self.agent2_name, self.agent1_name)
        )

        self.tasks = [task1, task2]

        # Wait for either task to complete (which indicates a connection closed or error)
        done, pending = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info("Bridge stopped")

    async def start(self):
        """Connect to both agents and start forwarding messages."""
        try:
            await self.connect()
            await self.run()
        finally:
            await self.disconnect()

    def stop(self):
        """Stop the bridge gracefully."""
        logger.info("Stopping bridge...")
        self.running = False
        for task in self.tasks:
            task.cancel()
