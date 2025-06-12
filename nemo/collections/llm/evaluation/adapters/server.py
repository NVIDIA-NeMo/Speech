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

"""Main server holding entry-point to all the interceptors.

The arch is as follows:


         ┌───────────────────────┐
         │                       │
         │ NVIDIA Eval Factory   │
         │                       │
         └───▲──────┬────────────┘
             │      │
     returns │      │
             │      │ calls
             │      │
             │      │
         ┌───┼──────┼──────────────────────────────────────────────────┐
         │   │      ▼                                                  │
         │ AdapterServer (@ localhost:<free port>)                     │
         │                                                             │
         │   ▲      │       chain of RequestInterceptors:              │
         │   │      │       flask.Request                              │
         │   │      │       is passed on the way up                    │
         │   │      │                                                  │   ┌──────────────────────┐
         │   │ ┌────▼───────────────────────────────────────────────┐  │   │                      │
         │   │ │intcptr_1─────►intcptr_2───►...───►intcptr_N────────┼──┼───►                      │
         │   │ │                     │                              │  │   │                      │
         │   │ └─────────────────────┼──────────────────────────────┘  │   │                      │
         │   │                       │(e.g. for caching interceptors,  │   │  upstream endpoint   │
         │   │                       │ this "shortcut" will happen)    │   │   with actual model  │
         │   │                       │                                 │   │                      │
         │   │                       └─────────────┐                   │   │                      │
         │   │                                     │                   │   │                      │
         │ ┌─┼─────────────────────────────────────▼────┐              │   │                      │
         │ │intcptr'_M◄──intcptr'_2◄──...◄───intcptr'_1 ◄──────────────┼───┤                      │
         │ └────────────────────────────────────────────┘              │   └──────────────────────┘
         │                                                             │
         │              Chain of ResponseInterceptors:                 │
         │              requests.Response is passed on the way down    │
         │                                                             │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘

In other words, interceptors are pieces of independent logic which should be
relatively easy to add separately.



"""

import logging as _original_logging
import multiprocessing
from typing import Optional, Tuple

import flask
import requests
import werkzeug.serving

from nemo.collections.llm.evaluation.api import AdapterConfig
from nemo.utils import logging

from .interceptors import (
    AdapterMetadata,
    AdapterRequest,
    AdapterResponse,
    EndpointInterceptor,
    RequestInterceptor,
    RequestLoggingInterceptor,
    ResponseInterceptor,
    ResponseLoggingInterceptor,
    ResponseReasoningInterceptor,
    SystemMessageInterceptor,
)
from .utils import find_free_port, wait_for_server


def create_server_process(
    adapter_config: AdapterConfig,
) -> Tuple[multiprocessing.Process, AdapterConfig]:
    """Create and start a server process, returning the process and the config.

    This makes sure that the factory function is not needing any complex serialization for
    multiprocessing.

    Args:
        api_url: The API URL the adapter will call
        adapter_config: Configuration for the adapter server

    Returns:
        Tuple of (process, adapter_config) where process is the running server process,
        and adapter_config is the configuration with port filled in.
    """

    adapter_config.local_port = (
        adapter_config.local_port if adapter_config.local_port is not None else find_free_port()
    )

    @staticmethod
    def create_server_factory(
        adapter_config: AdapterConfig,
    ) -> None:
        server = _AdapterServer(adapter_config=adapter_config)
        server.run()

    p = multiprocessing.Process(
        target=create_server_factory,
        args=(adapter_config,),
        kwargs={},
    )
    p.start()

    if not wait_for_server("localhost", adapter_config.local_port):
        p.terminate()
        raise ConnectionError(f"Could not wait till adapter server is up on localhost:{adapter_config.local_port}.")

    return (p, adapter_config)


class _AdapterServer:
    """Main server which serves on a local port and holds chain of interceptors"""

    adapter_host: str
    adapter_port: int

    request_interceptors: list[RequestInterceptor]
    response_interceptors: list[ResponseInterceptor]

    app: flask.Flask

    api_url: str

    def __init__(
        self,
        adapter_config: AdapterConfig,
    ):
        """
        Initializes the app, creates server and adds interceptors

        Args:
            adapter_config: should be obtained from the evaluation script, see `api.py`
        """

        self.request_interceptors = []
        self.response_interceptors = []

        self.app = flask.Flask(__name__)
        self.app.route("/", defaults={"path": ""}, methods=["POST"])(self._handler)
        self.app.route("/<path:path>", methods=["POST"])(self._handler)

        self.adapter_host: str = "localhost"
        assert adapter_config.local_port is not None
        self.adapter_port: int = adapter_config.local_port
        self.adapter_config = adapter_config

        logging.info(f"Using the following adapter config: {adapter_config}")

        self._build_interceptor_chains(
            use_reasoning=adapter_config.use_reasoning,
            end_reasoning_token=adapter_config.end_reasoning_token,
            custom_system_prompt=adapter_config.custom_system_prompt,
            max_logged_requests=adapter_config.max_logged_requests,
            max_logged_responses=adapter_config.max_logged_responses,
        )

    def _build_interceptor_chains(
        self,
        use_reasoning: bool,
        end_reasoning_token: str,
        custom_system_prompt: Optional[str],
        max_logged_requests: Optional[int],
        max_logged_responses: Optional[int],
    ):

        if custom_system_prompt:
            self.request_interceptors.append(SystemMessageInterceptor(new_system_message=custom_system_prompt))
        self.request_interceptors.append(RequestLoggingInterceptor(max_requests=max_logged_requests))
        self.request_interceptors.append(EndpointInterceptor(api_url=self.adapter_config.api_url))

        self.response_interceptors.append(ResponseLoggingInterceptor(max_responses=max_logged_responses))
        if use_reasoning:
            self.response_interceptors.append(ResponseReasoningInterceptor(end_reasoning_token=end_reasoning_token))

    def run(self) -> None:
        """Start the Flask server."""
        logging.info(f"Starting the evaluation adapter server at {self.adapter_host}:{self.adapter_port}...")
        # Below setting prevents from littering with
        # messges like `<...> INFO 127.0.0.1 - - [27/May/2025 05:58:11] "POST / HTTP/1.1" 200 -`
        _original_logging.getLogger('werkzeug').setLevel(logging.ERROR)
        werkzeug.serving.run_simple(
            hostname=self.adapter_host,
            port=self.adapter_port,
            application=self.app,
            threaded=True,
        )
        logging.info("Evaluation adapter server started")

    # The headers we don't want to let out
    _EXCLUDED_HEADERS = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]

    @classmethod
    def _process_response_headers(cls, response: requests.Response) -> list[tuple[str, str]]:
        """Process response headers, removing excluded ones."""
        return [(k, v) for k, v in response.headers.items() if k.lower() not in cls._EXCLUDED_HEADERS]

    def _handler(self, path: str) -> flask.Response:
        adapter_request = AdapterRequest(
            r=flask.request,
            meta=AdapterMetadata(),
        )
        adapter_response = None
        for interceptor in self.request_interceptors:
            output = interceptor.intercept_request(adapter_request)

            if isinstance(output, AdapterResponse):
                adapter_response = output
                break
            if isinstance(output, AdapterRequest):
                adapter_request = output

        assert adapter_response is not None, "There should be a response to process"

        for interceptor in self.response_interceptors:
            try:
                adapter_response = interceptor.intercept_response(adapter_response)
            except Exception as e:
                self._log_response_interceptor_error(interceptor, adapter_response, e)
                raise

        return flask.Response(
            response=adapter_response.r.content,
            status=adapter_response.r.status_code,
            headers=self._process_response_headers(adapter_response.r),
        )

    def _log_response_interceptor_error(
        self,
        interceptor: ResponseInterceptor,
        adapter_response: AdapterResponse,
        error: Exception,
    ) -> None:
        error_message = (
            f"❌ Error in Response Interceptor ❌\n"
            f"Interceptor: {interceptor.__class__.__name__}\n"
            f"Adapter Response Status Code: {adapter_response.r.status_code}\n"
            f"Adapter Response Status Text: {adapter_response.r.reason}\n"
            f"Adapter Response Content: {adapter_response.r.content.decode('utf-8', errors='ignore')}\n"
            f"Error Details: {repr(error)}\n"
        )
        logging.error(error_message)
