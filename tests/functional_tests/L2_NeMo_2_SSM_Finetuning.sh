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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/gpt/model/test_nemotronh.py \
    --num-nodes=1 \
    --devices=2 \
    --max-steps=20 \
    --val-check-interval=20 \
    --experiment-dir=/tmp/nlp_megatron_mamba_nemo-ux-mamba_cicd_test_sft/$RUN_ID \
    --ckpt-dir="/home/TestData/nlp/megatron_mamba/toy_nmh" \
    --vocab-file="/home/TestData/nlp/megatron_mamba/nm5_tokenizer/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json" \
    --dataset-dir="/home/TestData/nlp/megatron_mamba/toy_ssm_dataset/legal_pile_text_document" \
    --seq-length=512 \
    --restore-optimizer-from-ckpt \
    --hybrid-override-pattern="M-*" \
    --num-layers=3 \
    --tensor-parallel-size=2 \
    --pipeline-model-parallel-size=1 \
    --context-parallel-size=1 \
    --global-batch-size=8 \
    --micro-batch-size=1 \
    --model-size="4B" \
    --clip-grad 1 \
    --lr=0.0003 \
    --warmup-steps=0 \
    --no-wandb
