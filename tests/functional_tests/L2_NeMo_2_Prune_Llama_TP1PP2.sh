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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/test_hf_import.py --hf_model /home/TestData/nlp/megatron_llama/llama-ci-hf --output_path /tmp/nemo2_ckpt

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/llm/gpt_prune.py \
  --restore_path /tmp/nemo2_ckpt \
  --legacy_ckpt \
  --tokenizer gpt2 \
  --tp_size 1 \
  --pp_size 2 \
  --devices 2 \
  --seq_length 128 \
  --mbs 2 \
  --num_train_samples 32 \
  --data_paths 1.0 /home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document \
  --index_mapping_dir examples/nlp/language_modeling/gpt_index_mappings \
  --target_hidden_size 64 \
  --target_ffn_hidden_size 128 \
  --target_num_attention_heads 4 \
  --target_num_query_groups 4 \
  --target_num_layers 2 \
  --save_path /tmp/pruned-llama
