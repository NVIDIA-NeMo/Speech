#!/bin/bash

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

echo "Install latest AIS CLI"
AIS_CLI_URL=https://github.com/NVIDIA/aistore/releases/latest/download/ais-linux-amd64.tar.gz

echo "Download AIS CLI from ${AIS_CLI_URL}"
curl -LO ${AIS_CLI_URL}

echo "Extract"
tar -xzvf ais-linux-amd64.tar.gz

echo "Move to /usr/local/bin/"
mv ./ais /usr/local/bin/. 

echo "Cleanup"
rm ais-linux-amd64.tar.gz
