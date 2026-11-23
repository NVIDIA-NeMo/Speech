# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

scratch=$(mktemp -d "${NEMOTRON35_TEST_SCRATCH:-/tmp}/nemotron35-serving.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
export NEMO_HOME="$scratch/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_PLUGINS=nemo_speechlm
export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_VISIBLE_DEVICES=0
export COVERAGE_FILE="${NEMOTRON35_COVERAGE_FILE:-/workspace/.coverage}"

# Stage the fixture once so the engine and draft workers do their reads on local disk.
cp -r "${NEMOTRON35_SERVING_FIXTURE:-/home/TestData/speechlm/nemotron35-transcribe-ci-v1}" "$scratch/fixture"
cover=(coverage run --parallel-mode --source="$PWD/nemo")
"${cover[@]}" -m tests.functional_tests.nemotron35_transcribe_inference "$scratch/fixture" "$scratch/tokens.json"
"${cover[@]}" -m tests.functional_tests.nemotron35_transcribe_inference "$scratch/fixture" "$scratch/tokens.json" --mtp
