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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/llama3_pretraining.py \
    --seq-length 1024 \
    --devices=2 \
    --max-steps=6 \
    --early-stop=3 \
    --experiment-dir=/tmp/llm_tests/llama_pretrain_results \
    --data-path=/home/TestData/nlp/megatron_llama/data/rp2_sample_sentencepiece_preproc_text_document \
    --tokenizer-path=/home/TestData/nlp/megatron_llama/tokenizer.model \
    --index-mapping-dir=/tmp/llm_tests/llama_index_mappings \
    --ckpt-optim-fully-reshardable  # we change TP and CP in the subsequent call

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/llama3_pretraining.py \
    --seq-length 1024 \
    --devices=2 \
    --max-steps=6 \
    --experiment-dir=/tmp/llm_tests/llama_pretrain_results \
    --data-path=/home/TestData/nlp/megatron_llama/data/rp2_sample_sentencepiece_preproc_text_document \
    --tokenizer-path=/home/TestData/nlp/megatron_llama/tokenizer.model \
    --index-mapping-dir=/tmp/llm_tests/llama_index_mappings \
    --cp 1 --tp 2 --sp 1
