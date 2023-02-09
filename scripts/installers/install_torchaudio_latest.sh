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

TORCHAUDIO_REPO=https://github.com/pytorch/audio
# expected LATEST_RELEASE=release/*.**
LATEST_RELEASE=$(git -c 'versionsort.suffix=-' \
    ls-remote --exit-code --refs --sort='version:refname' --heads ${TORCHAUDIO_REPO} 'release/*.*' \
    | tail --lines=1 \
    | cut --delimiter='/' --fields=3,4)
# expected TORCHAUDIO_BUILD_VERSION=*.**.*
TORCHAUDIO_BUILD_VERSION=${LATEST_RELEASE:8:1}${PYTORCH_VERSION:1:5}

TORCH_MAJOR_VERSION=$(python3 -c "major_version = \"${PYTORCH_VERSION}\".split('.')[0]; print(major_version)")
TORCH_MINOR_VERSION=$(python3 -c "minor_version = \"${PYTORCH_VERSION}\".split('.')[1]; print(minor_version)")
TORCHAUDIO_MINOR_VERSION=$(python3 -c "minor_version = \"${LATEST_RELEASE}\".rsplit('.')[-1]; print(minor_version)")

if [[ $TORCH_MAJOR_VERSION -ne 1 ]]; then
    echo "WARNING: Pytorch major version different from 1 not supported"
fi

echo "Latest torchaudio release: ${LATEST_RELEASE:8:4}"
echo "Pytorch version: ${PYTORCH_VERSION:0:6}"
echo "Torchaudio build version: ${TORCHAUDIO_BUILD_VERSION}"

if [[ "$TORCH_MINOR_VERSION" -lt "$TORCHAUDIO_MINOR_VERSION" ]]; then
    # for old containers, we need to install matching torchaudio version
    INSTALL_BRANCH="release/0.${TORCH_MINOR_VERSION}"
else
    # for new containers use latest release
    INSTALL_BRANCH=${LATEST_RELEASE}
fi

echo "Installing torchaudio from branch: ${INSTALL_BRANCH}"

# we need parameterized to run torchaudio tests
# suppose that we do not have parameterized installed yet
pip install parameterized

# Build torchaudio and run MFCC test
git clone --depth 1 --branch ${INSTALL_BRANCH} https://github.com/pytorch/audio.git && \
cd audio && \
git submodule update --init --recursive && \
BUILD_SOX=1 BUILD_VERSION=${TORCHAUDIO_BUILD_VERSION} python setup.py install && \
cd .. && \
pytest -rs audio/test/torchaudio_unittest/transforms/torchscript_consistency_cpu_test.py -k 'test_MFCC' || \
{ echo "ERROR: Failed to install torchaudio!"; exit 1; };
# RNNT loss is built with CUDA, so checking it will suffice
# This test will be skipped if CUDA is not available (e.g. when building from docker)
pytest -rs audio/test/torchaudio_unittest/functional/torchscript_consistency_cuda_test.py -k 'test_rnnt_loss' || \
echo "WARNING: Failed to install torchaudio with CUDA support!";
rm -rf audio && \
echo "Torchaudio installed successfully!"
