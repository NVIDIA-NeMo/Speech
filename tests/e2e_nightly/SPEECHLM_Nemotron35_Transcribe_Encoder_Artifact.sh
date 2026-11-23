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

test_data=${NEMOTRON35_TEST_DATA:-/home/TestData}
encoder_fixture="$test_data/speechlm/nemotron35-transcribe-encoder-ci-v1"
audio_fixture="$test_data/speechlm/nemotron35-transcribe-ci-v1"
scratch=$(mktemp -d "${NEMOTRON35_TEST_SCRATCH:-/tmp}/nemotron35-encoder.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export NEMO_HOME="$scratch/cache"
export COVERAGE_FILE="${NEMOTRON35_COVERAGE_FILE:-/workspace/.coverage}"

# Stage the small synthetic encoder and reference once and verify on local disk.
cp "$encoder_fixture/"{parallel-expert-encoder.nemo,encoder-reference.safetensors} "$scratch/"
cp "$audio_fixture/"{short.wav,long.wav} "$scratch/"
(
    cd "$scratch"
    sha256sum --check <<'SHA256'
db9e2e58b018a9e0a5f09a03e388358daee9717245ea892a574975ce617d279e  parallel-expert-encoder.nemo
1a97f00db6e39a628dcd6b15fbe8443579754116b4b5634f36c11c61dc0eefb3  encoder-reference.safetensors
d1e0294eba40bd907d1aa493f296c4bfd78f5b9bc9f81a9b2b396046a5fed2e2  short.wav
792e52aeb746292451b9cfdccdfd0a754c517c721ea4d15d96cf47b024d9acd9  long.wav
SHA256
)
coverage run --parallel-mode --source="$PWD/nemo" -m tests.e2e_nightly.nemotron35_transcribe_encoder "$scratch"
