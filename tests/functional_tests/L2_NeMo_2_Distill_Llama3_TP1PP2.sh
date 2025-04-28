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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/llm/gpt_distillation.py \
  --name nemo2_llama_distill \
  --teacher_path /home/TestData/nemo2_ckpt/llama_68M_v4 \
  --student_path /home/TestData/nemo2_ckpt/llama_68M_v4 \
  --tp_size 1 \
  --cp_size 1 \
  --pp_size 2 \
  --devices 2 \
  --log_dir /tmp/distill_logs \
  --max_steps 5 \
  --gbs 4 \
  --mbs 1 \
  --data_paths 1.0 /home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document \
  --index_mapping_dir examples/nlp/language_modeling/gpt_index_mappings \
  --seq_length 2048 \
  --warmup_steps 1 \
  --val_check_interval 5 \
  --log_interval 5 \
  --limit_val_batches 2 \
  --legacy_ckpt
