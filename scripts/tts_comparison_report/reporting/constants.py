# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pathlib import Path


_ROOT: Path = Path(__file__).parent.parent

# Benchmark names supported by the comparison report pipeline.
SUPPORTED_BENCHMARK_NAMES: list[str] = [
    "libritts_seen",
    "libritts_test_clean",
    "riva_hard_digits",
    "riva_hard_letters",
    "riva_hard_money",
    "riva_hard_short",
    "vctk",
]

# Default width of tqdm progress bars in terminal columns.
TQDM_NCOLS: int = 80

# Random seed used for reproducible sampling of audio examples.
SEED: int = 42

# Number of decimal digits used when formatting p-values in statistical tests.
P_VAL_ROUND_DIGITS: int = 4

# Default lifetime of generated S3 presigned links in seconds (one year).
S3_LINK_EXPIRES_IN: int = 31536000

# Subdirectory inside the S3 report prefix used for uploaded audio files.
S3_AUDIO_DIR: str = "audio"

# Subdirectory inside the S3 report prefix used for uploaded plot images.
S3_IMAGES_DIR: str = "images"

# Directory containing Jinja templates used for report rendering.
TEMPLATES_DIR: Path = _ROOT / "templates"

# Fallback task id used when no real Jira ticket is provided.
DUMMY_TASK_ID: str = "NEMOTTS-0000"

# URL prefix used to construct clickable Jira ticket links in reports.
JIRA_TICKET_URL_PREFIX: str = "https://jirasw.nvidia.com/browse"
