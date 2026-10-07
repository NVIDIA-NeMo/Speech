# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Count tokens per data epoch for a manifest, and the batches it yields at a given batch_tokens.

Use it to set limit_train_batches strictly below the real batches/epoch (SKILL.md step 3);
estimating by hand over-counted ESC-50 by ~40% and the run trained without validating.
Counts audio tokens (duration / 0.08 s) + tokenized prompt + target + ~20 template tokens.

Usage: epoch_batches.py <manifest.json> <batch_tokens> [--tokenizer <ckpt>]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("batch_tokens", type=int)
ap.add_argument("--tokenizer", required=True, help="checkpoint directory (its tokenizer is used)")
a = ap.parse_args()
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.tokenizer)
with open(a.manifest) as f:
    rows = [json.loads(line) for line in f]
total = 0
for r in rows:
    total += int(float(r["duration"]) / 0.08) + len(tok.encode(r.get("context", ""))) + len(tok.encode(r["text"])) + 20
bpe = total / a.batch_tokens
print(
    f"{len(rows)} rows, {total} tokens/epoch ({total / len(rows):.0f}/row) -> {bpe:.1f} batches/epoch at "
    f"{a.batch_tokens}; suggest limit_train_batches <= {max(1, int(bpe * 0.8))}"
)
