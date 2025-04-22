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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/test_hf_import.py \
  --hf_model /home/TestData/nlp/megatron_llama/llama-ci-hf \
  --output_path /tmp/nemo2_ckpt

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/deploy/nlp/deploy_inframework_triton.py \
  --nemo_checkpoint /tmp/nemo2_ckpt \
  --triton_model_name llama \
  --num_gpus 2 \
  --tensor_parallelism_size 2 &

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/deploy/nlp/query_inframework.py \
  --model_name llama \
  --prompt "What is the color of a banana?" \
  --max_output_len 20
