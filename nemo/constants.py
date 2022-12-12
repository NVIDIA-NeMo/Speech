# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

NEMO_ENV_VARNAME_ENABLE_COLORING = "NEMO_ENABLE_COLORING"
NEMO_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR = "NEMO_REDIRECT_LOGS_TO_STDERR"
NEMO_ENV_VARNAME_TESTING = "NEMO_TESTING"  # Set to True to enable nemo.util.logging's debug mode
NEMO_ENV_VARNAME_VERSION = "NEMO_EXPM_VERSION"  # Used for nemo.utils.exp_manager versioning
NEMO_ENV_CACHE_DIR = "NEMO_CACHE_DIR"  # Used to change default nemo cache directory
NEMO_ENV_DATA_STORE_CACHE_DIR = "NEMO_DATA_STORE_CACHE_DIR"  # Used to change default nemo data store cache directory
NEMO_ENV_DATA_STORE_CACHE_SHARED = "NEMO_DATA_STORE_CACHE_SHARED"  # Shared among nodes (1) or not shared (0)
