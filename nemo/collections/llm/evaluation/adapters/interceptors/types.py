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


from abc import ABC, abstractmethod
from dataclasses import dataclass

import flask
import requests


@dataclass
class AdapterMetadata:
    """This is passed with the _whole_ chain from request back to response.

    Add here any useful information.
    """

    cache_hit: bool = False  # Whether there was a cache hit
    cache_key: str | None = None  # Cache key


@dataclass
class AdapterRequest:
    """Container for request data and metadata."""

    r: flask.Request
    meta: AdapterMetadata


@dataclass
class AdapterResponse:
    """Container for response data and metadata."""

    r: requests.Response
    meta: AdapterMetadata


class RequestInterceptor(ABC):
    """Interface for providing interception of requests."""

    @abstractmethod
    def intercept_request(
        self,
        req: AdapterRequest,
    ) -> AdapterRequest | AdapterResponse:
        """Function that will be called by `AdapterServer` on the way upstream.

        If the return type is `Request`, the chain will continue upstream.
        If the reruen type is `Response`, the `AdapterServer` will consider the
        chain finished and start the reverse chain of responses.
        """
        pass


class ResponseInterceptor(ABC):
    """Interface for providing interception of responses."""

    @abstractmethod
    def intercept_response(self, ar: AdapterResponse) -> AdapterResponse:
        """Intercept response on its way back to upastream, process it and send further downstream."""
        pass
