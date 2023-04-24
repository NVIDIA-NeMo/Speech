# syntax=docker/dockerfile:experimental

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.03-py3

# build an image that includes only the nemo dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `nemo-deps`)
FROM ${BASE_IMAGE} as nemo-deps

# dependency flags; should be declared after FROM
# torchaudio: not required by default
ARG REQUIRE_TORCHAUDIO=false
# k2: not required by default
ARG REQUIRE_K2=false
# ais cli: not required by default, install only if required
ARG REQUIRE_AIS_CLI=false

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
# libavdevice-dev rerquired for latest torchaudio
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
  libsndfile1 sox \
  libfreetype6 \
  swig \
  ffmpeg \
  libavdevice-dev && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/
# Install Megatron-core
RUN git clone https://github.com/NVIDIA/Megatron-LM.git && \
  cd Megatron-LM && \
  git checkout 3db2063b1ff992a971ba18f7101eecc9c4e90f03 && \
  pip install -e .

WORKDIR /tmp/
# TODO: Remove once this Apex commit (2/24/23) is included in PyTorch
# container
RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout 57057e2fcf1c084c0fcc818f55c0ff6ea1b24ae2 && \
  pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./

# uninstall stuff from base container
RUN pip3 uninstall -y sacrebleu torchtext

# build torchaudio
WORKDIR /tmp/torchaudio_build
COPY scripts/installers /tmp/torchaudio_build/scripts/installers/
RUN INSTALL_MSG=$(/bin/bash /tmp/torchaudio_build/scripts/installers/install_torchaudio_latest.sh); INSTALL_CODE=$?; \
  echo ${INSTALL_MSG}; \
  if [ ${INSTALL_CODE} -ne 0 ]; then \
  echo "torchaudio installation failed";  \
  if [ "${REQUIRE_TORCHAUDIO}" = true ]; then \
  exit ${INSTALL_CODE};  \
  else echo "Skipping failed torchaudio installation"; fi \
  else echo "torchaudio installed successfully"; fi

# install nemo dependencies
WORKDIR /tmp/nemo
COPY requirements .
RUN for f in $(ls requirements*.txt); do pip3 install --disable-pip-version-check --no-cache-dir -r $f; done

# install k2, skip if installation fails
COPY scripts /tmp/nemo/scripts/
RUN INSTALL_MSG=$(/bin/bash /tmp/nemo/scripts/speech_recognition/k2/setup.sh); INSTALL_CODE=$?; \
  echo ${INSTALL_MSG}; \
  if [ ${INSTALL_CODE} -ne 0 ]; then \
  echo "k2 installation failed";  \
  if [ "${REQUIRE_K2}" = true ]; then \
  exit ${INSTALL_CODE};  \
  else echo "Skipping failed k2 installation"; fi \
  else echo "k2 installed successfully"; fi

# copy nemo source into a scratch image
FROM scratch as nemo-src
COPY . .

# start building the final container
FROM nemo-deps as nemo
ARG NEMO_VERSION=1.18.0

# Check that NEMO_VERSION is set. Build will fail without this. Expose NEMO and base container
# version information as runtime environment variable for introspection purposes
RUN /usr/bin/test -n "$NEMO_VERSION" && \
  /bin/echo "export NEMO_VERSION=${NEMO_VERSION}" >> /root/.bashrc && \
  /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc

# Install NeMo
RUN --mount=from=nemo-src,target=/tmp/nemo cd /tmp/nemo && pip install ".[all]"

# Check install
RUN python -c "import nemo.collections.nlp as nemo_nlp" && \
  python -c "import nemo.collections.tts as nemo_tts" && \
  python -c "import nemo_text_processing.text_normalization as text_normalization"


# copy scripts/examples/tests into container for end user
WORKDIR /workspace/nemo
COPY scripts /workspace/nemo/scripts
COPY examples /workspace/nemo/examples
COPY tests /workspace/nemo/tests
COPY tutorials /workspace/nemo/tutorials
# COPY README.rst LICENSE /workspace/nemo/

RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
  chmod +x start-jupyter.sh

# If required, install AIS CLI
RUN if [ "${REQUIRE_AIS_CLI}" = true ]; then \
  INSTALL_MSG=$(/bin/bash scripts/installers/install_ais_cli_latest.sh); INSTALL_CODE=$?; \
  echo ${INSTALL_MSG}; \
  if [ ${INSTALL_CODE} -ne 0 ]; then \
  echo "AIS CLI installation failed"; \
  exit ${INSTALL_CODE}; \
  else echo "AIS CLI installed successfully"; fi \
  else echo "Skipping AIS CLI installation"; fi
