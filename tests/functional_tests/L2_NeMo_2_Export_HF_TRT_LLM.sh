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

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/export/nemo_export.py \
  --model_name test \
  --model_dir /tmp/trt_llm_model_dir/ \
  --model_type LlamaForCausalLM \
  --use_huggingface True \
  --checkpoint_dir /home/TestData/llm/models/llama3.2-1B-hf/ \
  --min_tps 1 \
  --test_deployment True \
  --debug
