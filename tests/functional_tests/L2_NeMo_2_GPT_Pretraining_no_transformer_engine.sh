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
pip uninstall -y apex ## TODO: remove when apex is no longer a dependency
pip uninstall -y transformer_engine

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_gpt_pretraining.py \
    --devices=2 \
    --max-steps=3 \
    --experiment-dir=tests/collections/llm/gpt_pretrain_results \
    --vocab-path=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    --merges-path=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    --data-path=/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document \
    --index-mapping-dir=tests/collections/llm/gpt_index_mappings \
    --no-masked-softmax-fusion

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_gpt_pretraining.py \
    --devices=2 \
    --max-steps=6 \
    --experiment-dir=tests/collections/llm/gpt_pretrain_results \
    --vocab-path=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    --merges-path=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    --data-path=/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document \
    --index-mapping-dir=tests/collections/llm/gpt_index_mappings \
    --no-masked-softmax-fusion
