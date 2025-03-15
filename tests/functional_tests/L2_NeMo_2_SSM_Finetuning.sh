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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/gpt/model/megatron_ssm_finetuning.py \
    --devices 1 \
    --max-steps 10 \
    --experiment-dir /tmp/nlp_megatron_mamba_nemo-ux-mamba_cicd_test_sft/$RUN_ID \
    --model-path /home/TestData/nlp/megatron_mamba/model_optim_rng.pt \
    --ckpt_load_strictness log_all
