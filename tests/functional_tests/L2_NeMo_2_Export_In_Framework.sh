# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/test_hf_import.py \
  --hf_model /home/TestData/nlp/megatron_llama/llama-ci-hf \
  --output_path /tmp/nemo2_ckpt

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/setup/data/create_sample_lambada.py \
  --output_file /tmp/lambada.json

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/export/nemo_export.py \
  --model_name test \
  --model_type llama \
  --checkpoint_dir /tmp/nemo2_ckpt \
  --min_tps 1 \
  --in_framework True \
  --test_deployment True \
  --run_accuracy True \
  --test_data_path /tmp/lambada.json \
  --accuracy_threshold 0.0 \
  --debug
