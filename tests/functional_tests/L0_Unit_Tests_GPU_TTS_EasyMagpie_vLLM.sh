#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION.
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

set -euo pipefail

CONTRACT_DIR=/workspace/.ci_artifacts/easymagpie-codec-contract
test -f "$CONTRACT_DIR/config.json"
test -f "$CONTRACT_DIR/model.safetensors"
test -f "$CONTRACT_DIR/contract.safetensors"
export EASYMAGPIE_CODEC_CONTRACT="$CONTRACT_DIR"

python3 -m pip install --no-build-isolation --no-deps -e tools/easymagpie_vllm_omni

CUDA_VISIBLE_DEVICES=0 coverage run \
    -a \
    --data-file=/workspace/.coverage \
    --source=/workspace/tools/easymagpie_vllm_omni \
    -m pytest \
    --confcutdir=tests/collections/tts/easymagpie_vllm_omni/serving \
    tests/collections/tts/easymagpie_vllm_omni/serving
