#!/usr/bin/env bash
set -ex

INSTALL_OPTION=${1:-"dev"}
HEAVY_DEPS=${HEAVY_DEPS:-false}

PIP=pip

${PIP} install -U ${PIP}

echo 'Uninstalling stuff'
${PIP} uninstall -y nemo_toolkit
${PIP} uninstall -y sacrebleu

# Kept for legacy purposes
${PIP} uninstall -y nemo_asr
${PIP} uninstall -y nemo_nlp
${PIP} uninstall -y nemo_tts

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TE_TAG=7d576ed25266a17a7b651f2c12e8498f67e0baea
export NEMO_RUN_TAG=34259bd3e752fef94045a9a019e4aaf62bd11ce2
export APEX_TAG=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
export CAUSAL_CONV_TAG=v1.2.2.post1
export MAMBA_TAG=v2.2.0
export MCORE_TAG=4dc8977167d71f86bdec47a60a98e85c4cfa0031
export NV_RESILIENCY_EXT_TAG=97aad77609d2e25ed38ac5c99f0c13f93c48464e

if [ -n "${NVIDIA_PYTORCH_VERSION}" ]; then
  echo "Installing NeMo in NVIDIA PyTorch container: ${NVIDIA_PYTORCH_VERSION}"

  echo "Will not install numba"
  ${PIP} install --no-build-isolation "apex @ git+https://github.com/NVIDIA/apex.git@${APEX_TAG}"

else
  if [ -n "${CONDA_PREFIX}" ]; then
    NUMBA_VERSION=0.57.1
    echo 'Installing numba=='${NUMBA_VERSION}
    conda install -y -c conda-forge numba==${NUMBA_VERSION}
  fi

  ${PIP} install torch
  ${PIP} install "apex @ git+https://github.com/NVIDIA/apex.git@${APEX_TAG}"

fi

DEPS=(
  "nvidia-modelopt[torch]~=0.21.0; sys_platform == 'linux'"
  "nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@${NEMO_RUN_TAG}"
  "git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_TAG}"
  "git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@${NV_RESILIENCY_EXT_TAG}"
  "onnxscript @ git+https://github.com/microsoft/onnxscript"
)

if [[ "$HEAVY_DEPS" == "TRUE" ]]; then
  ${PIP} install --no-cache-dir virtualenv &&
    virtualenv /opt/venv &&
    /opt/venv/bin/pip install --no-cache-dir --no-build-isolation \
      -r /workspace/requirements/requirements_vllm.txt \
      -r /workspace/requirements/requirements_infer.txt

  DEPS+=(
    "llama-index==0.10.43"
    "unstructured==0.14.9"
    "git+https://github.com/Dao-AILab/causal-conv1d.git@${CAUSAL_CONV_TAG}"
    "git+https://github.com/state-spaces/mamba.git@${MAMBA_TAG}"
    "triton==3.1.0"
  )

  pip install --no-cache-dir -r tools/ctc_segmentation/requirements.txt

  CURR=$(pwd)
  cd /opt
  git clone https://github.com/NVIDIA/Megatron-LM.git &&
    pushd Megatron-LM &&
    git checkout ${MCORE_TAG} &&
    pip install -e . &&
    popd

  git clone https://github.com/NVIDIA/TransformerEngine.git &&
    pushd TransformerEngine &&
    git checkout ${TE_TAG} &&
    git submodule update --init --recursive -q &&
    pip install -e . &&
    popd

  cd "$CURR"

fi

echo 'Installing dependencies of nemo'
${PIP} install --no-cache-dir --extra-index-url https://pypi.nvidia.com "${DEPS[@]}"

echo 'Installing nemo'
if [[ "$INSTALL_OPTION" == "dev" ]]; then
  ${PIP} install --editable ".[all]"

else
  rm -rf dist/ &&
    ${PIP} install build pytest-runner &&
    python -m build --no-isolation --wheel &&
    DIST_FILE=$(find ./dist -name "*.whl" | head -n 1) &&
    ${PIP} install "${DIST_FILE}[all]"

fi

echo 'All done!'
