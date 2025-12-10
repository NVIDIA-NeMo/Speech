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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/peft/lora_merge.py \
    --lora_checkpoint_path=/home/TestData/nemo2_ckpt/llama_lora_ci_checkpoint_v4/ \
    --output_path=/tmp/nemo2_lora_merge/$RUN_ID \
    --legacy_ckpt \
    --trust_remote_code
