# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

set -euo pipefail

scratch=$(mktemp -d "${NEMOTRON35_TEST_SCRATCH:-/tmp}/nemotron35-training.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
world_size=${NEMOTRON35_TEST_GPUS:-1} # Set to 2 on an NVLink runner to exercise DeepEP dispatch.
export NEMO_HOME="$scratch/cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export COVERAGE_FILE="${NEMOTRON35_COVERAGE_FILE:-/workspace/.coverage}"

python -m tests.functional_tests.nemotron35_transcribe_fixture \
    "$scratch" "${NEMOTRON35_TEST_DATA:-/home/TestData}" --world-size "$world_size"

torch_run=(torchrun --nnodes=1 --nproc-per-node="$world_size" --rdzv-backend=static
    --master-addr=127.0.0.1 --local-addr=127.0.0.1 --no-python)
cover=(coverage run --parallel-mode --source="$PWD/nemo")
train=(-m tests.functional_tests.nemotron35_transcribe_training "$scratch")

"${torch_run[@]}" "${cover[@]}" "${train[@]}" train
"${torch_run[@]}" "${cover[@]}" "${train[@]}" resume

"${torch_run[@]}" "${cover[@]}" examples/speechlm2/to_hf.py \
    class_path=nemo.collections.speechlm2.models.SALMAutomodel \
    "ckpt_path='$scratch/resumed/checkpoints/step=4.ckpt'" \
    "ckpt_config=$scratch/resumed/exp_config.yaml" \
    "output_dir=$scratch/export"
python -m tests.functional_tests.nemotron35_transcribe_training "$scratch" verify --world-size "$world_size"

# Maintainers can regenerate the small serving fixture from a validated training run.
if [[ -n "${NEMOTRON35_SAVE_SERVING_FIXTURE:-}" ]]; then
    mkdir "$NEMOTRON35_SAVE_SERVING_FIXTURE"
    cp -r "$scratch/export" "$NEMOTRON35_SAVE_SERVING_FIXTURE/model"
    cp "$scratch/short.wav" "$scratch/long.wav" "$NEMOTRON35_SAVE_SERVING_FIXTURE/"
fi
