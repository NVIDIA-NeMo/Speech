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


from typing import final

import requests

from .types import AdapterRequest, AdapterResponse, RequestInterceptor


@final
class EndpointInterceptor(RequestInterceptor):
    """Intercepts requests and forwards them to a specified API endpoint."""

    api_url: str

    def __init__(self, api_url: str):
        self.api_url = api_url

    @final
    def intercept_request(self, ar: AdapterRequest) -> AdapterResponse:
        # This is a final interceptor, we'll need the flask_request and api
        resp = AdapterResponse(
            r=requests.request(
                method=ar.r.method,
                url=self.api_url,
                headers={k: v for k, v in ar.r.headers if k.lower() != "host"},
                json=ar.r.json,
                cookies=ar.r.cookies,
                allow_redirects=False,
            ),
            meta=ar.meta,
        )
        return resp
