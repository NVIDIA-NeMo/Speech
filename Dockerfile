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

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.12-py3

# build an image that includes only the nemo dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `nemo-deps`)
FROM ${BASE_IMAGE} as nemo-deps

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    libsndfile1 sox \
    libfreetype6 \
    swig \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/

# uninstall stuff from base container
RUN pip3 uninstall -y sacrebleu torchtext

# build torchaudio
WORKDIR /tmp/torchaudio_build
COPY scripts/installers /tmp/torchaudio_build/scripts/installers/
RUN /bin/bash /tmp/torchaudio_build/scripts/installers/install_torchaudio_latest.sh

# install nemo dependencies
WORKDIR /tmp/nemo
COPY requirements .
RUN for f in $(ls requirements*.txt); do pip3 install --disable-pip-version-check --no-cache-dir -r $f; done

# install pynini
COPY nemo_text_processing/install_pynini.sh /tmp/nemo/
RUN /bin/bash /tmp/nemo/install_pynini.sh

# install k2, skip if installation fails
COPY scripts /tmp/nemo/scripts/
RUN /bin/bash /tmp/nemo/scripts/speech_recognition/k2/setup.sh || exit 0

# copy nemo source into a scratch image
FROM scratch as nemo-src
COPY . .

# start building the final container
FROM nemo-deps as nemo
ARG NEMO_VERSION=1.15.0

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

# TODO: Update to newer numba 0.56.0RC1 for 22.03 container if possible
# install pinned numba version
# RUN conda install -c conda-forge numba==0.54.1

# Pinned to numba==0.53.1 to avoid bug in training with num_workers > 0
# The bug still exists with PTL 1.8.4, this is just a temporary workaround.
RUN pip install numba==0.53.1

# copy scripts/examples/tests into container for end user
WORKDIR /workspace/nemo
COPY scripts /workspace/nemo/scripts
COPY examples /workspace/nemo/examples
COPY tests /workspace/nemo/tests
COPY tutorials /workspace/nemo/tutorials
# COPY README.rst LICENSE /workspace/nemo/

RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
    chmod +x start-jupyter.sh

# Prepare AIS CLI
ARG AIS_VERSION=v1.3.15
ARG AIS_BIN=https://github.com/NVIDIA/aistore/releases/download/${AIS_VERSION}/ais-linux-amd64.tar.gz
RUN curl -LO ${AIS_BIN} && tar -xzvf ais-linux-amd64.tar.gz && mv ./ais /usr/local/bin/. && rm ais-linux-amd64.tar.gz
