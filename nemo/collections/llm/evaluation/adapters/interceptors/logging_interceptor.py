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


import threading
from typing import Optional, final

import requests

from nemo.utils import logging

from .types import AdapterRequest, AdapterResponse, RequestInterceptor, ResponseInterceptor


def _get_safe_headers(headers: dict[str, str]) -> dict[str, str]:
    """Create a copy of headers with authorization redacted."""
    safe_headers = dict(headers)
    for header in safe_headers:
        if header.lower() == 'authorization':
            safe_headers[header] = '[REDACTED]'
    return safe_headers


@final
class RequestLoggingInterceptor(RequestInterceptor):
    """Intercepts and logs incoming requests with configurable limits."""

    _max_requests: Optional[int]
    _logged_requests: int
    # Interceptors might executed concurrenlty.
    _lock: threading.Lock

    def __init__(self, max_requests: Optional[int] = 5):
        """
        Initialize the request logging interceptor.

        Args:
            output_dir: Directory to store logs in
            max_requests: Maximum number of requests to log. If None, all requests will be logged.
            log_failed_requests: Whether to log failed request-response pairs (status code >= 400)
        """
        self._max_requests = max_requests
        self._logged_requests = 0
        self._lock = threading.Lock()
        logging.info(
            "Evaluation logging adapter will log "
            f"{self._max_requests if self._max_requests is not None else 'all'} requests"
        )

    def _log_request(self, ar: AdapterRequest) -> None:
        """Helper method to log request details"""
        try:
            # Try to parse as JSON first
            payload = ar.r.get_json()
            log_data = {
                "request": {
                    "method": ar.r.method,
                    "url": ar.r.url,
                    "headers": _get_safe_headers(ar.r.headers),
                    "payload": payload,
                }
            }
            logging.info(f"Request with data: {log_data}")

        except Exception:
            # If JSON parsing fails, log raw data
            log_data = {
                "request": {
                    "method": ar.r.method,
                    "url": ar.r.url,
                    "headers": _get_safe_headers(ar.r.headers),
                    "raw_data": ar.r.get_data().decode('utf-8', errors='ignore'),
                }
            }
            logging.warning("Invalid request JSON, logging a request raw data")
            logging.info(f"Request with raw data: {log_data}")

    @final
    def intercept_request(self, ar: AdapterRequest) -> AdapterRequest:
        # Check if we should log this request
        if self._max_requests is not None and self._logged_requests >= self._max_requests:
            return ar

        self._log_request(ar)

        with self._lock:
            self._logged_requests += 1

        return ar


@final
class ResponseLoggingInterceptor(ResponseInterceptor):
    """Intercepts and logs outgoing responses with configurable limits."""

    _max_responses: Optional[int]
    _logged_responses: int
    # Interceptors might executed concurrenlty.
    _lock: threading.Lock

    def __init__(self, max_responses: Optional[int] = 5):
        """
        Initialize the response logging interceptor.

        Args:
            use_response_logging: Whether to log full responses to the console
            max_responses: max responses to log, if `None` all of them are logged
        """
        self._max_responses = max_responses
        self._logged_responses = 0
        self._lock = threading.Lock()
        logging.info(
            "Evaluation logging adapter will log "
            f"{self._max_responses if self._max_responses is not None else 'all'} responses"
        )

    @final
    def intercept_response(self, ar: AdapterResponse) -> AdapterResponse:
        # Check if we should log this
        if self._max_responses is not None and self._logged_responses >= self._max_responses:
            return ar

        try:
            payload = ar.r.json()

            log_data = {
                "response": {"status_code": ar.r.status_code, "payload": payload, "cache_hit": ar.meta.cache_hit}
            }
            logging.info(f"Response with data: {log_data}")

        except requests.exceptions.JSONDecodeError:
            # For non-JSON responses, only log if response logging is enabled
            log_data = {
                "response": {
                    "status_code": ar.r.status_code,
                    "raw_content": ar.r.content.decode('utf-8', errors='ignore'),
                }
            }
            logging.warning("Invalid response JSON, logging response raw data")
            logging.info(f"Response with raw data: {log_data}")

        with self._lock:
            self._logged_responses += 1

        return ar
