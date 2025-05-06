# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

mkdir -p /tmp/llm_tests/llama_local_ckpt_results

coverage run \
    --data-file=/workspace/.coverage \
    --source=/workspace/nemo \
    tests/collections/llm/test_local_ckpt.py \
        --log-dir=/tmp/llm_tests/llama_local_ckpt_results \
        --devices=2 \
        --async-save
