#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test for scripts/installers/Dockerfile.speech_translation_vllm.
# Builds the image from a pinned revision (never a moving branch) and checks that the interpreter and
# the packages needed for streaming speech translation are importable without installing anything else.
#
# Usage: scripts/installers/test_speech_translation_vllm_image.sh [CHECKOUT_SHA] [GIT_URL]
set -euo pipefail

CHECKOUT="${1:-$(git rev-parse HEAD)}"
GIT_URL="${2:-https://github.com/NVIDIA-NeMo/Speech.git}"
CONTEXT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="nemo-speech-translation-vllm-smoke:${CHECKOUT:0:12}"

docker build -f "${CONTEXT}/Dockerfile.speech_translation_vllm" \
  --build-arg GIT_URL="${GIT_URL}" --build-arg CHECKOUT="${CHECKOUT}" -t "${IMAGE}" "${CONTEXT}"

docker run --rm -i "${IMAGE}" python - <<'PY'
import sys

assert sys.version_info >= (3, 12), sys.version
# Triton compiles a CUDA helper against the interpreter headers the first time a pipeline runs.
import os
import sysconfig

python_h = os.path.join(sysconfig.get_paths()["include"], "Python.h")
assert os.path.exists(python_h), f"missing {python_h}: install python3-dev"
import torch  # noqa: F401
import vllm  # noqa: F401
from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder  # noqa: F401
from nemo.collections.asr.inference.nmt.llm_translator import LLMTranslator  # noqa: F401

print("speech translation image smoke test: OK")
PY
