#!/usr/bin/env bash
set -e

INSTALL_OPTION=${1:-"dev"}

PIP=pip

${PIP} install -U ${PIP}

echo 'Uninstalling stuff'
${PIP} uninstall -y nemo_toolkit
${PIP} uninstall -y sacrebleu

# Kept for legacy purposes
${PIP} uninstall -y nemo_asr
${PIP} uninstall -y nemo_nlp
${PIP} uninstall -y nemo_tts

if [ -n "${NVIDIA_PYTORCH_VERSION}" ]; then
  echo 'Installing NeMo in NVIDIA PyTorch container:' "${NVIDIA_PYTORCH_VERSION}" 'so will not install numba'
else
  if [ -n "${CONDA_PREFIX}" ]; then
    NUMBA_VERSION=0.57.1
    echo 'Installing numba=='${NUMBA_VERSION}
    conda install -y -c conda-forge numba==${NUMBA_VERSION}
  fi
fi

echo 'Installing nemo'
if [[ "$INSTALL_OPTION" == "dev" ]]; then
    ${PIP} install --editable ".[all]"
else
    rm -rf dist/
    ${PIP} install build pytest-runner
    python -m build --no-isolation --wheel
    DIST_FILE=$(find ./dist -name "*.whl" | head -n 1)
    ${PIP} install "${DIST_FILE}[all]"
fi

echo 'All done!'
