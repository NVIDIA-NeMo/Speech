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

scratch=$(mktemp -d "${NEMOTRON35_TEST_SCRATCH:-/tmp}/nemotron35-training.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
export CUDA_VISIBLE_DEVICES=0
export NEMO_HOME="$scratch/cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export COVERAGE_FILE="${NEMOTRON35_COVERAGE_FILE:-/workspace/.coverage}"

python -m pytest --noconftest -v --basetemp="$scratch/pytest" -o timeout=600 \
    tests/functional_tests/nemotron35_transcribe_training.py "$@"
