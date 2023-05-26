# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


K2_INSTALLATION_MESSAGE = (
    "Could not import `k2`.\n"
    "Please install k2 in one of the following ways:\n"
    "1) (recommended) Run `bash scripts/speech_recognition/k2/setup.sh`\n"
    "2) Use any approach from https://k2-fsa.github.io/k2/installation/index.html "
    "if your your cuda and pytorch versions are supported.\n"
    "It is advised to always install k2 using setup.sh only, "
    "as different versions of k2 may not interact with the NeMo code as expected."
)
