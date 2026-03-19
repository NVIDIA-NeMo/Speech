# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Precision related constants
BIG_EPSILON = 1e-5
SMALL_EPSILON = 1e-10
ROUND_PRECISION = 9

# ASR Preprocessing related constants
LOG_MEL_ZERO = -16.635

# Punctuation related constants
POST_WORD_PUNCTUATION = set(".,?")
PRE_WORD_PUNCTUATION = set("¿")
SEP_REPLACEABLE_PUNCTUATION = set("-_")
SENTENCEPIECE_UNDERSCORE = "▁"

# ITN related constants
DEFAULT_SEMIOTIC_CLASS = "name"

# Default output directory name
DEFAULT_OUTPUT_DIR_NAME = "jsons"
