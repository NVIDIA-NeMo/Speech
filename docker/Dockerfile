# syntax=docker/dockerfile:1-labs

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

# This CI Dockerfile supports CUDA 13 and CUDA 12 from a single BASE_IMAGE
# build arg. The default is the recommended CUDA 13 image:
#   nvcr.io/nvidia/cuda-dl-base:26.04-cuda13.2-devel-ubuntu24.04
# The current recommended CUDA 12 image is:
#   nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04
#
# The build derives CUDA_FLAVOR internally from BASE_IMAGE by matching
# "cuda13" or "cuda12" in the image tag. That flavor selects the matching uv
# extra (cu13 or cu12) and CUDA Python package include path. If BASE_IMAGE does
# not contain either token, the build fails early.
#
# Example CUDA 13 H100+ build:
#   docker buildx build -f docker/Dockerfile \
#     --build-arg GPU_TARGET=h100plus .
# Example CUDA 12 A100 build:
#   docker buildx build -f docker/Dockerfile \
#     --build-arg BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04 \
#     --build-arg GPU_TARGET=a100 .
#
# GPU_TARGET controls compiled Automodel dependency tuning. "h100plus" builds
# for SM90/SM100/SM120 and includes H100+ features such as DeepEP and
# flash-attn-4. "a100" builds only SM80 and uses an A100-specific DeepEP patch
# to avoid unsupported newer-GPU/NVSHMEM build paths. This keeps CI images
# smaller and avoids compiling kernels for architectures a target image cannot
# use.
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.04-cuda13.2-devel-ubuntu24.04
FROM ${BASE_IMAGE} AS base-image
ARG BASE_IMAGE
ARG UV_VERSION=0.11.14

ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_OFFLINE=0
ENV HYDRA_FULL_ERROR=1
ENV PYTHONUNBUFFERED=1
ENV UV_PYTHON=3.13
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV UV_NO_CACHE=1
ENV UV_NO_PROGRESS=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/root/.local/bin:${VIRTUAL_ENV}/bin:${PATH}"

RUN <<"EOF" bash -ex
apt-get update
apt-get install -y \
    bc \
    curl \
    git \
    libsndfile1 \
    libsox-fmt-all \
    sox
apt-get clean
EOF

RUN <<"EOF" bash -ex
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
uv --version
EOF

RUN <<"EOF" bash -euxo pipefail
case "${BASE_IMAGE}" in
    *cuda12*) cuda_flavor=cu12 ;;
    *cuda13*) cuda_flavor=cu13 ;;
    *)
        echo "Cannot derive CUDA flavor from BASE_IMAGE='${BASE_IMAGE}'. Expected image tag containing 'cuda12' or 'cuda13'."
        exit 1
        ;;
esac
cuda_major_minor="$(sed -n 's/.*cuda\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' <<<"${BASE_IMAGE}")"
if [[ -z "${cuda_major_minor}" ]]; then
    echo "Cannot derive CUDA major.minor from BASE_IMAGE='${BASE_IMAGE}'. Expected image tag containing e.g. 'cuda12.9' or 'cuda13.2'."
    exit 1
fi
cat >/usr/local/bin/nemo-cuda-flavor <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
echo "${cuda_flavor}"
SCRIPT
cat >/usr/local/bin/nemo-install-cuda-python <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cuda_major_minor="${cuda_major_minor}"
cuda_major="\${cuda_major_minor%%.*}"
cuda_minor="\${cuda_major_minor#*.}"
cuda_next_minor="\$((cuda_minor + 1))"
uv pip install \
    "cuda-bindings>=\${cuda_major_minor}.0,<\${cuda_major}.\${cuda_next_minor}" \
    "cuda-python>=\${cuda_major_minor}.0,<\${cuda_major}.\${cuda_next_minor}"
SCRIPT
chmod +x /usr/local/bin/nemo-cuda-flavor /usr/local/bin/nemo-install-cuda-python
echo "Derived CUDA flavor: ${cuda_flavor}"
echo "Derived CUDA Python major.minor: ${cuda_major_minor}"
EOF

WORKDIR /workspace
COPY pyproject.toml uv.lock /workspace/
COPY nemo/__init__.py nemo/package_info.py /workspace/nemo/
RUN <<"EOF" bash -ex
cuda_flavor="$(nemo-cuda-flavor)"
uv sync --link-mode copy --locked --extra all --extra "${cuda_flavor}" --group test
nemo-install-cuda-python
EOF

RUN <<"EOF" bash -ex
# Container-only runtime utilities. Keep these out of pyproject.toml so they do
# not become NeMo package dependencies.
uv pip install \
    dill \
    orjson

case "$(nemo-cuda-flavor)" in
    cu12) torchcodec_index=https://download.pytorch.org/whl/cu126 ;;
    cu13) torchcodec_index=https://download.pytorch.org/whl/cu132 ;;
esac
uv pip install --index-url "${torchcodec_index}" torchcodec
EOF

FROM base-image AS automodel-deps
ARG GPU_TARGET=h100plus
ARG DEEPEP_REPO=https://github.com/deepseek-ai/DeepEP.git
ARG DEEPEP_REF=v1.2.1
ARG FLASH_ATTN_MAX_JOBS=8
ARG FLASH_ATTN_NVCC_THREADS=1

ENV PIP_NO_CACHE_DIR=1

COPY external/patches/deepep_v1_a100_disable_nvshmem.patch /tmp/patches/
RUN <<"EOF" bash -euxo pipefail
apt-get update
apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    ninja-build \
    patch
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

RUN <<"EOF" bash -euxo pipefail
cat >/usr/local/bin/automodel-build-env <<'SCRIPT'
case "${GPU_TARGET}" in
    h100plus)
        export DEEPEP_TORCH_CUDA_ARCH_LIST="9.0+PTX"
        export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"
        export NVTE_CUDA_ARCHS="90;100;120"
        export FLASH_ATTN_CUDA_ARCHS="90;100;120"
        export CMAKE_CUDA_ARCHITECTURES="90;100;120"
        ;;
    a100)
        export DEEPEP_TORCH_CUDA_ARCH_LIST="8.0"
        export TORCH_CUDA_ARCH_LIST="8.0"
        export NVTE_CUDA_ARCHS="80"
        export FLASH_ATTN_CUDA_ARCHS="80"
        export CMAKE_CUDA_ARCHITECTURES="80"
        export DISABLE_SM90_FEATURES=1
        ;;
    *)
        echo "Unsupported GPU_TARGET='${GPU_TARGET}'. Expected 'h100plus' or 'a100'."
        exit 1
        ;;
esac

CUDA_FLAVOR="$(nemo-cuda-flavor)"
AUTOMODEL_CCCL_INCLUDES="/usr/local/cuda/include/cccl"
PYTHON_CCCL_INCLUDE="${VIRTUAL_ENV}/lib/python${UV_PYTHON}/site-packages/nvidia/${CUDA_FLAVOR}/include/cccl"
if [[ -d "${PYTHON_CCCL_INCLUDE}" ]]; then
    AUTOMODEL_CCCL_INCLUDES="${AUTOMODEL_CCCL_INCLUDES}:${PYTHON_CCCL_INCLUDE}"
fi
export CPATH="${AUTOMODEL_CCCL_INCLUDES}${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${AUTOMODEL_CCCL_INCLUDES}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS="${FLASH_ATTN_MAX_JOBS}"
export NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}"
SCRIPT
mkdir -p /opt/automodel-src
EOF

COPY external/patches/flash_attn_4_sm120_disable_tma_o.patch /tmp/patches/
RUN <<"EOF" bash -euxo pipefail
source /usr/local/bin/automodel-build-env
uv pip install --upgrade \
    cmake \
    ninja \
    nvidia-mathdx \
    packaging \
    pip \
    psutil \
    pybind11 \
    setuptools \
    setuptools-scm \
    wheel

automodel_extra=compiled
if [[ "${GPU_TARGET}" == "h100plus" ]]; then
    automodel_extra=compiled
elif [[ "${GPU_TARGET}" == "a100" ]]; then
    automodel_extra=compiled-a100
fi
cuda_flavor="$(nemo-cuda-flavor)"
uv sync \
    --inexact \
    --link-mode copy \
    --locked \
    --extra all \
    --extra "${cuda_flavor}" \
    --extra "${automodel_extra}" \
    --group test
nemo-install-cuda-python

if [[ "${GPU_TARGET}" == "a100" ]]; then
    git clone "${DEEPEP_REPO}" /opt/automodel-src/DeepEP
    git -C /opt/automodel-src/DeepEP checkout -f "${DEEPEP_REF}"
    patch -d /opt/automodel-src/DeepEP -p1 < /tmp/patches/deepep_v1_a100_disable_nvshmem.patch
    TORCH_CUDA_ARCH_LIST="${DEEPEP_TORCH_CUDA_ARCH_LIST}" \
        uv pip install \
        --no-build-isolation \
        --no-deps \
        /opt/automodel-src/DeepEP
fi

if [[ "${GPU_TARGET}" == "h100plus" ]]; then
    # flash-attn-4 requires apache-tvm-ffi 0.1.11, while mamba-ssm
    # currently constrains the solved environment to apache-tvm-ffi<=0.1.9.
    cutlass_packages=(
        "nvidia-cutlass-dsl==4.5.2"
        "nvidia-cutlass-dsl-libs-base==4.5.2"
    )
    if [[ "$(nemo-cuda-flavor)" == "cu13" ]]; then
        cutlass_packages+=("nvidia-cutlass-dsl-libs-cu13==4.5.2")
    fi
    uv pip install \
        --no-deps \
        "apache-tvm-ffi==0.1.11" \
        "${cutlass_packages[@]}" \
        "quack-kernels==0.5.0" \
        "torch-c-dlpack-ext==0.1.5"

    mkdir -p /opt/automodel-src/flash-attn-4 /tmp/flash-attn-4
    python -m pip download \
        --no-binary=flash-attn-4 \
        --no-deps \
        --dest /tmp/flash-attn-4 \
        "flash-attn-4==4.0.0b15"
    tar -xf /tmp/flash-attn-4/flash_attn_4-*.tar.gz -C /opt/automodel-src/flash-attn-4 --strip-components=1
    patch -d /opt/automodel-src/flash-attn-4 -p1 < /tmp/patches/flash_attn_4_sm120_disable_tma_o.patch

    MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" \
    NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
    uv pip install \
        --no-build-isolation \
        --no-deps \
        /opt/automodel-src/flash-attn-4
fi
EOF

FROM base-image
COPY --from=automodel-deps /opt/venv /opt/venv

ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
ARG RC_DATE=00.00
ARG TARGETARCH

ARG INSTALL_FFMPEG=false
RUN <<"EOF" bash -ex
if [ "${INSTALL_FFMPEG}" = "true" ]; then
    apt-get update
    apt-get install -y ffmpeg
    apt-get clean
    rm -rf /var/lib/apt/lists/*
fi
EOF

COPY nemo /workspace/nemo

# NOTICES.txt file points to where the OSS source code is archived
RUN echo "This distribution includes open source which is archived at the following URL: https://opensource.nvidia.com/oss/teams/nvidia/nemo/${RC_DATE}:linux-${TARGETARCH}/index.html" > NOTICES.txt && \
    echo "For further inquiries or assistance, contact us at oss-requests@nvidia.com" >> NOTICES.txt
