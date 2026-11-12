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
# Sourced by the L0_Unit_Tests_GPU_* scripts.
#
# Pull requests run only the tests marked `@pytest.mark.gpu`; every other test runs in the L0_Unit_Tests_CPU_* jobs.
# The scheduled (nightly) run executes the whole unit-test suite on GPU and fails if a test allocates CUDA memory
# without the marker (`--check-gpu-markers`), so missing markers are caught within a day.
if [[ "${CI_EVENT_NAME:-}" == "schedule" ]]; then
    GPU_TEST_SELECTION=(-m "not pleasefixme" --check-gpu-markers)
else
    GPU_TEST_SELECTION=(-m "gpu and not pleasefixme")
fi
