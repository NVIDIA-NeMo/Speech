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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_t5_pretraining.py \
    --devices=2 \
    --max-steps=3 \
    --experiment-dir=tests/collections/llm/t5_pretrain_results/$RUN_ID \
    --data-path=/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document \
    --index-mapping-dir=tests/collections/llm/t5_index_mappings/$RUN_ID

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_t5_pretraining.py \
    --devices=2 \
    --max-steps=6 \
    --experiment-dir=tests/collections/llm/t5_pretrain_results/$RUN_ID \
    --data-path=/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document \
    --index-mapping-dir=tests/collections/llm/t5_index_mappings/$RUN_ID
