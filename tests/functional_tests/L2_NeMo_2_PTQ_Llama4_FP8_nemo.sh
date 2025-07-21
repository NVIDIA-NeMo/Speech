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
NEMO_ORI_PATH=/home/TestData/llm/models/llama_4_16e_toy/

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/vlm/llama4/llama4_ptq.py -ctp 1 -nc ${NEMO_ORI_PATH} -algo fp8 -out /tmp/nemo2_llama4_ptq_ckpt -bs 1 --export_format nemo --model_id hf-internal-testing/tiny-random-llama4 --legacy_ckpt

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/vlm/llama4/llama4_generate.py --local_model_path /tmp/nemo2_llama4_ptq_ckpt --tp 1 --model_id hf-internal-testing/tiny-random-llama4