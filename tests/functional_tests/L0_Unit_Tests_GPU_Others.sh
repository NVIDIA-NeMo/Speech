# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES.
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
# Every unit test outside ASR and SpeechLM2. SpeechLM2 runs in its own job because some of its test modules set
# torch's default device to CUDA at import time, which would affect every other test collected in the same session.
source tests/functional_tests/_gpu_unit_test_selection.sh
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NEMO_NUMBA_MINVER=0.53 CUDA_VISIBLE_DEVICES=0 coverage run -a --data-file=/workspace/.coverage --source=/workspace/ -m pytest \
    "${GPU_TEST_SELECTION[@]}" --with_downloads \
    --ignore=tests/collections/asr \
    --ignore=tests/collections/llm \
    --ignore=tests/collections/multimodal \
    --ignore=tests/collections/nlp \
    --ignore=tests/collections/speechlm2 \
    --ignore=tests/collections/tts/easymagpie_vllm_omni/serving \
    --ignore=tests/export \
    --ignore=tests/deploy \
    --ignore=tests/functional_tests
