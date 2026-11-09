# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
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

"""Prefetch Magpie release checkpoints into the HF cache before offline model-support tests.

Run with ``HF_HUB_OFFLINE=0 python -m tests.e2e_nightly.magpie_release_checkpoints``.
The downloader and subsequent tests must use the same HF_HOME (CI mounts it under TestData).
"""

from huggingface_hub import hf_hub_download

MODEL_NAME = "nvidia/magpie_tts_multilingual_357m"
HF_NEMO_FILE = "magpie_tts_multilingual_357m.nemo"
MODEL_RELEASES = ("v2512", "v2602", "v2607")


def prefetch_release_checkpoints():
    """Cache every release used by the restore test, propagating download failures."""
    for revision in MODEL_RELEASES:
        print(f"Prefetching {MODEL_NAME}@{revision}/{HF_NEMO_FILE}", flush=True)
        hf_hub_download(repo_id=MODEL_NAME, filename=HF_NEMO_FILE, revision=revision)


if __name__ == "__main__":
    prefetch_release_checkpoints()
