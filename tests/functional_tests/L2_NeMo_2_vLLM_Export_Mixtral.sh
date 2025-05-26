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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/setup/models/create_hf_model.py \
  --model_name_or_path /home/TestData/hf/Mixtral-8x7B-Instruct-v0.1 \
  --output_dir /tmp/mixtral_tiny_hf \
  --config_updates "{\"num_hidden_layers\": 2, \"hidden_size\": 128, \"intermediate_size\": 448, \"num_attention_heads\": 4, \"num_key_value_heads\": 2, \"head_dim\": 32, \"num_local_experts\": 4}" &&
  coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/test_hf_import.py --hf_model /tmp/mixtral_tiny_hf --model MixtralModel --config MixtralConfig8x7B --output_path /tmp/mixtral_tiny_nemo2 &&
  /opt/venv/bin/coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/export/nemo_export.py \
    --min_tps 1 \
    --max_tps 1 \
    --use_vllm True \
    --model_type mixtral \
    --max_output_len 128 \
    --test_deployment True \
    --model_name nemo2_ckpt \
    --model_dir /tmp/vllm_from_nemo2 \
    --checkpoint_dir /tmp/mixtral_tiny_nemo2
