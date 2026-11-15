#!/usr/bin/env bash
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

# Smoke-test that the CI image's uv and Python run as root and as a non-root user.
#
# Usage: docker/check_runtime_users.sh <image>
#
# uv and its managed CPython must live outside /root (mode 0700): /opt/venv/bin/python
# symlinks into UV_PYTHON_INSTALL_DIR, so a root-only install denies every other UID.
set -euo pipefail

IMAGE="${1:?usage: $0 <image>}"

for user in 0:0 1000:1000; do
    echo "--- ${IMAGE} as ${user}"
    docker run --rm --user "${user}" --entrypoint /bin/bash "${IMAGE}" -euo pipefail -c '
        test "$(command -v uv)" = /usr/local/bin/uv
        test "$(command -v python)" = /opt/venv/bin/python
        uv --version
        uvx --version
        python -c "import sys, dill; assert sys.version_info[:2] == (3, 13), sys.version; assert sys.prefix == \"/opt/venv\", sys.prefix; print(sys.executable, sys.version.split()[0])"
    '
done
