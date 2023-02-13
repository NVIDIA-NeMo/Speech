# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import threading

import requests

headers = {"Content-Type": "application/json"}

lock = threading.Lock()

__all__ = ["request_data", "lock"]


def request_data(data, ip='localhost', port=None):
    resp = requests.put(f'http://{ip}:{port}/knn', data=json.dumps(data), headers=headers)
    return resp.json()


def text_generation(data, ip='localhost', port=None):
    resp = requests.put(f'http://{ip}:{port}/generate', data=json.dumps(data), headers=headers)
    return resp.json()


def convert_retrieved_to_md(retrieved):
    output_str = '<table><tr><th>Query</th><th>Retrieved Doc</th></tr>'
    for item in retrieved:
        output_str += f'<tr><td rowspan="{len(item["neighbors"])}">{item["query"]}</td>'
        for i, neighbor in enumerate(item['neighbors']):
            if i == 0:
                output_str += f"<td>{neighbor}</td></tr>"
            else:
                output_str += f"<tr><td>{neighbor}</td></tr>"
    output_str += '</table>'
    return output_str
