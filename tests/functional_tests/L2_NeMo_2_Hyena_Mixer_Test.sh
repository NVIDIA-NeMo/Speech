#!/bin/bash

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

set -e

# Set environment variables for the test
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Create log directory
RUN_ID=hyena_mixer_test_$(date +%Y%m%d_%H%M%S)
LOG_DIR=/tmp/nemo2_hyena_results/$RUN_ID
mkdir -p $LOG_DIR

# Set visible devices
export CUDA_VISIBLE_DEVICES=0,1

# Run the test with torchrun via coverage
echo "Running SE Hyena Mixer CP test with torchrun..."
coverage run -a \
    --data-file=/workspace/.coverage \
    --source=/workspace/nemo \
    -m torch.distributed.run \
    --nproc_per_node=2 \
    tests/collections/llm/gpt/model/test_hyena_mixer_cp.py \
    --context_parallel_size=2 \
    --operator_type=hyena_short_conv \
    --log_dir=$LOG_DIR

echo "Running MR Hyena Mixer CP test with torchrun..."
coverage run -a \
    --data-file=/workspace/.coverage \
    --source=/workspace/nemo \
    -m torch.distributed.run \
    --nproc_per_node=2 \
    tests/collections/llm/gpt/model/test_hyena_mixer_cp.py \
    --context_parallel_size=2 \
    --operator_type=hyena_medium_conv \
    --log_dir=$LOG_DIR

echo "Running LI Hyena Mixer CP test with torchrun..."
coverage run -a \
    --data-file=/workspace/.coverage \
    --source=/workspace/nemo \
    -m torch.distributed.run \
    --nproc_per_node=2 \
    tests/collections/llm/gpt/model/test_hyena_mixer_cp.py \
    --context_parallel_size=2 \
    --operator_type=hyena \
    --log_dir=$LOG_DIR

# Check exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Hyena Mixer CP tests completed successfully"
else
    echo "Hyena Mixer CP tests failed with status $STATUS"
    exit $STATUS
fi
