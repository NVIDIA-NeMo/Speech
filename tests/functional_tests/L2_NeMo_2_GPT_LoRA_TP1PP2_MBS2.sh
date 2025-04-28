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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/gpt_finetuning.py \
    --restore_path /home/TestData/nemo2_ckpt/llama_68M_v4 \
    --devices 2 \
    --max_steps 3 \
    --experiment_dir /tmp/nemo2_gpt_finetune/$RUN_ID \
    --peft lora \
    --tp_size 1 \
    --pp_size 2 \
    --mbs 2

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/gpt_finetuning.py \
    --restore_path /home/TestData/nemo2_ckpt/llama_68M_v4 \
    --devices 2 \
    --max_steps 6 \
    --experiment_dir /tmp/nemo2_gpt_finetune/$RUN_ID \
    --peft lora \
    --tp_size 1 \
    --pp_size 2 \
    --mbs 2
