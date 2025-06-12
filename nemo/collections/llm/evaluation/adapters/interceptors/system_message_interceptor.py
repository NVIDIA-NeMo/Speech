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


import json
from typing import final

from flask import Request

from nemo.utils import logging

from .types import AdapterRequest, RequestInterceptor


@final
class SystemMessageInterceptor(RequestInterceptor):
    """Adapter for injecting a custom system message into the payload"""

    _new_system_message: str

    def __init__(self, new_system_message: str):
        self._new_system_message = new_system_message
        logging.info(
            f"Evaluation adapter will inject system prompt with message:\n\"\"\"\n{self._new_system_message}\n\"\"\""
        )

    @final
    def intercept_request(self, ar: AdapterRequest) -> AdapterRequest:
        new_data = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": self._new_system_message},
                    *json.loads(ar.r.get_data())["messages"],
                ],
                **{k: v for k, v in json.loads(ar.r.get_data()).items() if k != "messages"},
            }
        )

        new_request = Request.from_values(
            path=ar.r.path, headers=dict(ar.r.headers), data=new_data, method=ar.r.method
        )
        return AdapterRequest(
            r=new_request,
            meta=ar.meta,
        )
