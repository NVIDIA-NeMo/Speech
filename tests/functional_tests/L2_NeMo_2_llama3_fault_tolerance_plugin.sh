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
mkdir -p /tmp/llm_tests/llama_pretrain_results \
    export FAULT_TOL_CFG_PATH="/tmp/llm_tests/llama_pretrain_results/sample_job_ft_cfg.yml"
export FAULT_TOL_FINISHED_FLAG_FILE="/tmp/llm_tests/llama_pretrain_results/sample_job_finished_flag"
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/test_fault_nvrx.py \
    --devices=2 \
    --crash-step=16 \
    --experiment-dir=/tmp/llm_tests/llama_pretrain_results \
    --data-path=/home/TestData/nlp/megatron_llama/data/rp2_sample_sentencepiece_preproc_text_document \
    --tokenizer-path=/home/TestData/nlp/megatron_llama/tokenizer.model \
    --index-mapping-dir=/tmp/llm_tests/llama_index_mappings \
    2>&1 | tee /tmp/llm_tests/llama_pretrain_results/run.log
