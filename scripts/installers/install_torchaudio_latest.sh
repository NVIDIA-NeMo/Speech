#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# Torch and torchaudio versions must match. Othervise, there will be no CUDA support.
# See https://github.com/pytorch/audio/blob/f0bc00c980012badea8db011f84a0e9ef33ba6c1/README.md?plain=1#L66

DEPENDENCIES_INSTALL_CMD="apt update && apt install -y ffmpeg sox libavdevice-dev"

read -r -d '' INFO_MESSAGE << EOM
INFO: This script is supposed to be used when building a docker container using Dockerfile in NeMo.
Use the script only for compiling torchaudio from scratch with a Non-Standard PyTorch version (e.g., 2.1.0a0+32f93b1)
For the release PyTorch version (e.g., 2.1.0), use 'pip install torchaudio' instead.
If running stand-alone, install dependencies first: '${DEPENDENCIES_INSTALL_CMD}'
EOM

echo "$INFO_MESSAGE"

for lib in libavdevice sox; do
  if ! grep -q ${lib} <<< "$(ldconfig -p)"; then
    echo "ERROR: ${lib} not found. Install dependencies before running the script: '${DEPENDENCIES_INSTALL_CMD}'"
    exit 1
  fi
done

if ! command -v ffmpeg &> /dev/null; then
  echo "ERROR: ffmpeg not found. Install dependencies before running the script: '${DEPENDENCIES_INSTALL_CMD}'"
  exit 1
fi

TORCHAUDIO_REPO=https://github.com/pytorch/audio
# expected LATEST_RELEASE=release/*.**
LATEST_RELEASE=$(git -c 'versionsort.suffix=-' \
    ls-remote --exit-code --refs --sort='version:refname' --heads ${TORCHAUDIO_REPO} 'release/*.*' \
    | tail --lines=1 \
    | cut -d '/' -f 3,4)
TORCHAUDIO_LATEST_MAJOR_VERSION=$(python3 -c "major_version = (\"${LATEST_RELEASE}\".split('/')[-1]).split('.')[0]; print(major_version)")
TORCHAUDIO_LATEST_MINOR_VERSION=$(python3 -c "minor_version = \"${LATEST_RELEASE}\".rsplit('.')[-1]; print(minor_version)")

# avoid checking PYTORCH_VERSION variable, not available everywhere
TORCH_FULL_VERSION=$(python3 -c "import torch; print(torch.__version__)")
TORCH_MAIN_VERSION=$(python3 -c "import torch, re; print(re.search(r'(\d+\.?)+', torch.__version__).group(0))")
TORCH_MAJOR_VERSION=$(python3 -c "major_version = \"${TORCH_MAIN_VERSION}\".split('.')[0]; print(major_version)")
TORCH_MINOR_VERSION=$(python3 -c "minor_version = \"${TORCH_MAIN_VERSION}\".split('.')[1]; print(minor_version)")
TORCH_FIX_VERSION=$(python3 -c "minor_version = \"${TORCH_MAIN_VERSION}\".split('.')[2]; print(minor_version)")


echo "Latest torchaudio release: ${TORCHAUDIO_LATEST_MAJOR_VERSION}.${TORCHAUDIO_LATEST_MINOR_VERSION}"
echo "Pytorch version: ${TORCH_MAIN_VERSION:0:6}"

if [[ $TORCH_MAJOR_VERSION -eq 1 ]]; then
  if [[ $TORCH_MINOR_VERSION -le 13 ]]; then
    INSTALL_BRANCH="release/0.${TORCH_MINOR_VERSION}"
  else
    # fix for PyTorch 1.14 (no official release)
    INSTALL_BRANCH="release/2.0"
  fi
  TORCHAUDIO_MAJOR_VERSION=0
else  # version 2 expected
  TORCHAUDIO_MAJOR_VERSION=${TORCH_MAJOR_VERSION}
  INSTALL_BRANCH="release/${TORCH_MAJOR_VERSION}.${TORCH_MINOR_VERSION}"
fi


# check if install branch exists
if [[ $(git ls-remote --heads ${TORCHAUDIO_REPO} ${INSTALL_BRANCH} | wc -l) -eq 0 ]]
then
  echo "Branch ${INSTALL_BRANCH} does not exist in torchaudio repo. Using latest release."
  INSTALL_BRANCH=${LATEST_RELEASE}
fi

# expected TORCHAUDIO_BUILD_VERSION=*.**.*
TORCHAUDIO_BUILD_VERSION="${TORCHAUDIO_MAJOR_VERSION}.${TORCH_MINOR_VERSION}.${TORCH_FIX_VERSION}"

echo "Torchaudio build version: ${TORCHAUDIO_BUILD_VERSION}"
echo "Installing torchaudio from branch: ${INSTALL_BRANCH}"

# we need parameterized to run torchaudio tests
# suppose that we do not have parameterized installed yet
pip install parameterized

# Build torchaudio and run MFCC test
git clone --depth 1 --branch ${INSTALL_BRANCH} https://github.com/pytorch/audio.git && \
cd audio && \
git submodule update --init --recursive && \
USE_FFMPEG=1 BUILD_SOX=1 BUILD_VERSION=${TORCHAUDIO_BUILD_VERSION} python setup.py install && \
cd .. && \
pytest -rs audio/test/torchaudio_unittest/transforms/torchscript_consistency_cpu_test.py -k 'test_MFCC' || \
{ echo "ERROR: Failed to install torchaudio!"; exit 1; };
# RNNT loss is built with CUDA, so checking it will suffice
# This test will be skipped if CUDA is not available (e.g. when building from docker)
pytest -rs audio/test/torchaudio_unittest/functional/torchscript_consistency_cuda_test.py -k 'test_rnnt_loss' || \
echo "WARNING: Failed to install torchaudio with CUDA support!";
rm -rf audio && \
echo "Torchaudio installed successfully!"
