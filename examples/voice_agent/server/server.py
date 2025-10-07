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
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv(override=True)

from bot_websocket_server import run_bot_websocket_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    yield  # Run app


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, path: str = "/ws"):
    raise NotImplementedError("FastAPI websocket endpoint is not implemented")


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    print("Received /connect request")
    server_mode = os.getenv("WEBSOCKET_SERVER", "websocket_server")
    if server_mode == "websocket_server":
        # Use the host that the client connected to (from the request)
        server_host = request.url.hostname or request.headers.get("host", "").split(":")[0]
        ws_url = f"ws://{server_host}:8765"
    else:
        ws_url = "ws://localhost:7860/ws"
    print(f"Returning WebSocket URL: {ws_url}")
    return {"ws_url": ws_url}


async def main():
    server_mode = os.getenv("WEBSOCKET_SERVER", "websocket_server")
    tasks = []
    try:
        if server_mode == "websocket_server":
            tasks.append(run_bot_websocket_server())
        else:
            raise ValueError(f"Invalid server mode: {server_mode}")
        config = uvicorn.Config(app, host="0.0.0.0", port=7860)
        server = uvicorn.Server(config)
        tasks.append(server.serve())

        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Tasks cancelled (probably due to shutdown).")


if __name__ == "__main__":
    asyncio.run(main())
