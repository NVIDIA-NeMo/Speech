# Copyright (c) 2025, NVIDIA CORPORATION.
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

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/workspace:${PYTHONPATH:-}

python - <<'PY'
import torch

if torch.cuda.device_count() < 2:
    raise SystemExit("Distributed OOMptimizer functional test requires at least 2 visible CUDA devices.")
PY

CONFIG_PATH=/tmp/distributed_oomptimizer_tiny.yaml
PROBE_LOG_DIR=/tmp/distributed_oomptimizer_probes_${RUN_ID:-manual}
rm -rf "${PROBE_LOG_DIR}"

python - <<'PY'
from pathlib import Path

Path("/tmp/distributed_oomptimizer_tiny.yaml").write_text(
    """
model:
  vocab_size: 32
  scratch_mb_per_sample: 96
trainer:
  devices: 2
  accelerator: gpu
  num_nodes: 1
  logger: false
  enable_checkpointing: false
  use_distributed_sampler: false
  max_steps: 1
  limit_train_batches: 1
  limit_val_batches: 0
  num_sanity_val_steps: 0
""".lstrip()
)
PY

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speechlm2/distributed_oomptimizer.py \
  -c "${CONFIG_PATH}" \
  -m tests.collections.speechlm2.distributed_oomptimizer_model.TinyDistributedOOMptimizerModel \
  -b "[0.05]" \
  -r 2.0 \
  -s 2 \
  -t 0.25 \
  --memory-fraction 0.01 \
  --nproc-per-node 2 \
  --probe-log-dir "${PROBE_LOG_DIR}" \
  --probe-timeout-seconds 180 \
  --probe-memory-reclaim-timeout-seconds 0

find "${PROBE_LOG_DIR}" -name 'probe_*.jsonl' -print -quit | grep -q .
