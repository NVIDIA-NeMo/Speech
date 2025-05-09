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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/gpt/model/test_hyena.py \
    --mock-data \
    --experiment-dir=tests/collections/llm/hyena_pretrain_results/$RUN_ID \
    --model-size=7b_nv \
    --num-layers=4 \
    --hybrid-override-pattern=SDH* \
    --no-activation-checkpointing \
    --add-bias-output \
    --max-steps=5 \
    --warmup-steps=1 \
    --micro-batch-size=2 \
    --global-batch-size=4 \
    --tensor-parallel-size=2 \
    --no-wandb \
    --seq-length=128 \
    --hidden-dropout=0.01 \
    --attention-dropout=0.01 \
    --devices=2
