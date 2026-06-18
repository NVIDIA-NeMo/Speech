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

CONFIG_PATH=/tmp/distributed_oomptimizer_transformer.yaml
PROBE_LOG_DIR=/tmp/distributed_oomptimizer_probes_${RUN_ID:-manual}
rm -rf "${PROBE_LOG_DIR}"

python - <<'PY'
from pathlib import Path

Path("/tmp/distributed_oomptimizer_transformer.yaml").write_text(
    """
model:
  vocab_size: 64
  sample_rate: 32
  frame_stride: 4
  hidden_size: 128
  num_heads: 4
  ffn_hidden_size: 256
  dropout: 0.0
  activation_reserve_mb_per_frame: 7
  max_activation_reserve_frames: 160
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
  -m tests.collections.speechlm2.distributed_oomptimizer_model.SingleBlockDistributedOOMptimizerModel \
  -b "[12.0,14.0,16.0,18.0,20.0]" \
  -r 2.0 \
  -s 8 \
  -t 0.25 \
  --memory-fraction 0.5 \
  --nproc-per-node 2 \
  --probe-log-dir "${PROBE_LOG_DIR}" \
  --probe-timeout-seconds 180 \
  --probe-memory-reclaim-timeout-seconds 0

PROBE_LOG_DIR="${PROBE_LOG_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

log_dir = Path(os.environ["PROBE_LOG_DIR"])
records = []
for path in sorted(log_dir.glob("probe_*.jsonl")):
    with path.open() as f:
        records.extend(json.loads(line) for line in f if line.strip())

if not records:
    raise SystemExit(f"No distributed OOMptimizer probe records found in {log_dir}.")

buckets = {record["bucket"] for record in records}
if len(buckets) != 5:
    raise SystemExit(f"Expected probe records for 5 buckets, found {len(buckets)}: {sorted(buckets)}")

if not any(record["status"] == "memory_target" for record in records):
    raise SystemExit("Expected at least one probe to stop at the requested memory target.")
PY
