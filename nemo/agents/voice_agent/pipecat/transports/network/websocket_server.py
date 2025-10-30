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
from typing import Optional

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, InputAudioRawFrame, StartFrame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.network.websocket_server import (
    WebsocketServerCallbacks,
    WebsocketServerOutputTransport,
    WebsocketServerParams,
)

from nemo.agents.voice_agent.pipecat.transports.base_input import BaseInputTransport
from nemo.agents.voice_agent.pipecat.transports.base_transport import TransportParams

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class WebsocketServerParams(TransportParams):
    """Configuration parameters for WebSocket server transport.

    Parameters:
        add_wav_header: Whether to add WAV headers to audio frames.
        serializer: Frame serializer for message encoding/decoding.
        session_timeout: Timeout in seconds for client sessions.
    """

    add_wav_header: bool = False
    serializer: Optional[FrameSerializer] = None
    session_timeout: Optional[int] = None


class WebsocketServerInputTransport(BaseInputTransport):
    """WebSocket server input transport for receiving client data.

    Handles incoming WebSocket connections, message processing, and client
    session management including timeout monitoring and connection lifecycle.
    """

    def __init__(
        self,
        transport: BaseTransport,
        host: str,
        port: int,
        params: WebsocketServerParams,
        callbacks: WebsocketServerCallbacks,
        **kwargs,
    ):
        """Initialize the WebSocket server input transport.

        Args:
            transport: The parent transport instance.
            host: Host address to bind the WebSocket server to.
            port: Port number to bind the WebSocket server to.
            params: WebSocket server configuration parameters.
            callbacks: Callback functions for WebSocket events.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        self._server_task = None

        # This task will monitor the websocket connection periodically.
        self._monitor_task = None

        self._stop_server_event = asyncio.Event()

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the WebSocket server and initialize components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.serializer:
            await self._params.serializer.setup(frame)
        if not self._server_task:
            self._server_task = self.create_task(self._server_task_handler())
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the WebSocket server and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        self._stop_server_event.set()
        if self._monitor_task:
            await self.cancel_task(self._monitor_task)
            self._monitor_task = None
        if self._server_task:
            await self.wait_for_task(self._server_task)
            self._server_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the WebSocket server and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        if self._monitor_task:
            await self.cancel_task(self._monitor_task)
            self._monitor_task = None
        if self._server_task:
            await self.cancel_task(self._server_task)
            self._server_task = None

    async def cleanup(self):
        """Cleanup resources and parent transport."""
        await super().cleanup()
        await self._transport.cleanup()

    async def _server_task_handler(self):
        """Handle WebSocket server startup and client connections."""
        logger.info(f"Starting websocket server on {self._host}:{self._port}")
        async with websockets.serve(self._client_handler, self._host, self._port) as server:
            await self._callbacks.on_websocket_ready()
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket: websockets.WebSocketServerProtocol, path: Optional[str] = None):
        """Handle individual client connections and message processing."""
        logger.info(f"New client connection from {websocket.remote_address}")
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client connected, using new connection")

        self._websocket = websocket

        # Notify
        await self._callbacks.on_client_connected(websocket)

        # Create a task to monitor the websocket connection
        if not self._monitor_task and self._params.session_timeout:
            self._monitor_task = self.create_task(self._monitor_websocket(websocket, self._params.session_timeout))

        # Handle incoming messages
        try:
            async for message in websocket:
                if not self._params.serializer:
                    continue

                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await self._websocket.close()
        self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")

    async def _monitor_websocket(self, websocket: websockets.WebSocketServerProtocol, session_timeout: int):
        """Monitor WebSocket connection for session timeout."""
        try:
            await asyncio.sleep(session_timeout)
            if not websocket.closed:
                await self._callbacks.on_session_timeout(websocket)
        except asyncio.CancelledError:
            logger.info(f"Monitoring task cancelled for: {websocket.remote_address}")
            raise


class WebsocketServerTransport(BaseTransport):
    """WebSocket server transport for bidirectional real-time communication.

    Provides a complete WebSocket server implementation with separate input and
    output transports, client connection management, and event handling for
    real-time audio and data streaming applications.
    """

    def __init__(
        self,
        params: WebsocketServerParams,
        host: str = "localhost",
        port: int = 8765,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the WebSocket server transport.

        Args:
            params: WebSocket server configuration parameters.
            host: Host address to bind the server to. Defaults to "localhost".
            port: Port number to bind the server to. Defaults to 8765.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
            on_websocket_ready=self._on_websocket_ready,
        )
        self._input: Optional[WebsocketServerInputTransport] = None
        self._output: Optional[WebsocketServerOutputTransport] = None
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")
        self._register_event_handler("on_websocket_ready")

    def input(self) -> WebsocketServerInputTransport:
        """Get the input transport for receiving client data.

        Returns:
            The WebSocket server input transport instance.
        """
        if not self._input:
            self._input = WebsocketServerInputTransport(
                self, self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> WebsocketServerOutputTransport:
        """Get the output transport for sending data to clients.

        Returns:
            The WebSocket server output transport instance.
        """
        if not self._output:
            self._output = WebsocketServerOutputTransport(self, self._params, name=self._output_name)
        return self._output

    async def _on_client_connected(self, websocket):
        """Handle client connection events."""
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnection events."""
        if self._output:
            await self._output.set_client_connection(None)
            await self._call_event_handler("on_client_disconnected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_session_timeout(self, websocket):
        """Handle client session timeout events."""
        await self._call_event_handler("on_session_timeout", websocket)

    async def _on_websocket_ready(self):
        """Handle WebSocket server ready events."""
        await self._call_event_handler("on_websocket_ready")
