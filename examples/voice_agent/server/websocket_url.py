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

from urllib.parse import urlsplit


def _normalize_websocket_scheme(scheme: str) -> str:
    scheme = scheme.strip().lower()
    if scheme not in {"ws", "wss"}:
        raise ValueError("WEBSOCKET_SCHEME must be either 'ws' or 'wss'")
    return scheme


def _normalize_websocket_host(host: str) -> str:
    host = host.strip()
    if not host:
        raise ValueError("SERVER_PUBLIC_HOST must not be empty")
    if "://" in host:
        parsed_host = urlsplit(host).hostname
        if not parsed_host:
            raise ValueError("SERVER_PUBLIC_HOST must include a host name")
        host = parsed_host
    return host


def build_websocket_url(host: str, port: int, scheme: str = "ws") -> str:
    """Build the client-facing WebSocket URL from trusted server configuration."""
    scheme = _normalize_websocket_scheme(scheme)
    host = _normalize_websocket_host(host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"
