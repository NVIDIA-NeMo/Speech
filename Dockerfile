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

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3

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
# libavdevice-dev required for latest torchaudio
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
  libsndfile1 sox \
  libfreetype6 \
  swig \
  ffmpeg \
  libavdevice-dev && \
  rm -rf /var/lib/apt/lists/*

# libtool, ... , libgts-dev are required for graphviz
# graphviz is required for k2 and pynini visualization
RUN apt-get update && \
  apt-get install -y \
  libtool \
  libltdl-dev \
  automake \
  autoconf \
  bison \
  flex \
  tcl \
  ghostscript \
  libgd-dev \
  fontconfig \
  libcairo2-dev \
  libpango1.0-dev \
  libgts-dev && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/
# Install megatron core, this can be removed once 0.3 pip package is released
# We leave it here in case we need to work off of a specific commit in main
RUN git clone https://github.com/NVIDIA/Megatron-LM.git && \
  cd Megatron-LM && \
  git checkout 36e9b6bf3d8034b10c9bbd9fc357c2df2bd1515c && \
  pip install .

# Performance optimizations for distributed optimizer: https://github.com/NVIDIA/apex/pull/1771
RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout f058162b215791b15507bb542f22ccfde49c872d && \
  pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

# Transformer Engine 1.2.0
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git fetch origin da30634a6c9ccdbb6c587b6c93b1860e4b038204 && \
  git checkout FETCH_HEAD && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

WORKDIR /tmp/

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

COPY scripts /tmp/nemo/scripts/
# install correct graphviz version (k2 and pynini visualization tool), skip if installation fails
RUN INSTALL_MSG=$(/bin/bash /tmp/nemo/scripts/installers/install_graphviz.sh --docker); INSTALL_CODE=$?; \
  echo ${INSTALL_MSG}; \
  if [ ${INSTALL_CODE} -ne 0 ]; then \
  echo "graphviz installation failed";  \
  if [ "${REQUIRE_K2}" = true ]; then \
  exit ${INSTALL_CODE};  \
  else echo "Skipping failed graphviz installation"; fi \
  else echo "graphviz installed successfully"; fi

# install k2, skip if installation fails
COPY scripts /tmp/nemo/scripts/
RUN INSTALL_MSG=$(/bin/bash /tmp/nemo/scripts/installers/install_k2.sh); INSTALL_CODE=$?; \
  echo ${INSTALL_MSG}; \
  if [ ${INSTALL_CODE} -ne 0 ]; then \
  echo "k2 installation failed";  \
  if [ "${REQUIRE_K2}" = true ]; then \
  exit ${INSTALL_CODE};  \
  else echo "Skipping failed k2 installation"; fi \
  else echo "k2 installed successfully"; fi

# install nemo dependencies
WORKDIR /tmp/nemo
ENV LHOTSE_REQUIRE_TORCHAUDIO=0
COPY requirements .
RUN for f in $(ls requirements*.txt); do pip3 install --disable-pip-version-check --no-cache-dir -r $f; done

# install flash attention
RUN pip install flash-attn
# install numba for latest containers
RUN pip install numba>=0.57.1
# install ammo
RUN pip install nvidia-ammo~=0.7.0 --extra-index-url https://pypi.nvidia.com --no-cache-dir

# copy nemo source into a scratch image
FROM scratch as nemo-src
COPY . .

# start building the final container
FROM nemo-deps as nemo
ARG NEMO_VERSION=1.23.0

# Check that NEMO_VERSION is set. Build will fail without this. Expose NEMO and base container
# version information as runtime environment variable for introspection purposes
RUN /usr/bin/test -n "$NEMO_VERSION" && \
  /bin/echo "export NEMO_VERSION=${NEMO_VERSION}" >> /root/.bashrc && \
  /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc

# Install NeMo
RUN --mount=from=nemo-src,target=/tmp/nemo,rw cd /tmp/nemo && pip install ".[all]"

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
