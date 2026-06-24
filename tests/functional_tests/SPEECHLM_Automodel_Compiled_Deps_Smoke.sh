# Copyright (c) 2026, NVIDIA CORPORATION.
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

SMOKE_TEST=tests/functional_tests/speechlm_automodel_compiled_deps_smoke.py
export SMOKE_TEST

timeout 300s torchrun --nproc-per-node 2 --no-python python -c \
  '
import os
import pytest

args = [os.environ["SMOKE_TEST"], "-m", "integration and not pleasefixme", "--with_downloads"]
os._exit(pytest.main(args))
'
