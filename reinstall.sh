#!/usr/bin/env bash
set -ex

# List of all supported libraries (update this list when adding new libraries)
# This also defines the order in which they will be installed by --libraries "all"
ALL_LIBRARIES=(
  "te"
  "apex"
  "mcore"
  "nemo"
)

INSTALL_OPTION=${1:-dev}
HEAVY_DEPS=${HEAVY_DEPS:-false}
CURR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
WHEELS_DIR=${WHEELS_DIR:-'/tmp/wheels'}

PIP=pip
${PIP} install -U ${PIP}

mcore() {
  local mode="$1"
  export MAMBA_FORCE_BUILD=TRUE
  export MAMBA_TAG=v2.2.0
  export CAUSAL_CONV1D_FORCE_BUILD=TRUE
  export CAUSAL_CONV_TAG=v1.2.2.post1

  CAUSAL_CONV1D_DIR="/opt/causal-conv1d" &&
    if [ ! -d "$CAUSAL_CONV1D_DIR/.git" ]; then
      rm -rf "$CAUSAL_CONV1D_DIR" &&
        cd $(dirname "$CAUSAL_CONV1D_DIR") &&
        git clone https://github.com/Dao-AILab/$(basename $CAUSAL_CONV1D_DIR).git
    fi &&
    pushd $CAUSAL_CONV1D_DIR &&
    git checkout -f $CAUSAL_CONV_TAG &&
    popd

  MAMBA_DIR="/opt/mamba" &&
    if [ ! -d "$MAMBA_DIR/.git" ]; then
      rm -rf "$MAMBA_DIR" &&
        cd $(dirname "$MAMBA_DIR") &&
        git clone https://github.com/state-spaces/$(basename $MAMBA_DIR).git
    fi &&
    pushd $MAMBA_DIR &&
    git checkout -f $MAMBA_TAG &&
    popd

  MLM_DIR="/opt/Megatron-LM" &&
    if [ ! -d "$MLM_DIR/.git" ]; then
      rm -rf "$MLM_DIR" &&
        cd $(dirname "$MLM_DIR") &&
        git clone ${MLM_REPO}
    fi &&
    pushd $MLM_DIR &&
    git checkout -f $MLM_TAG &&
    popd

  if [[ "$mode" == "build" ]]; then
    pip wheel --no-deps --wheel-dir $WHEELS_DIR/mcore/ $MAMBA_DIR
    pip wheel --no-deps --wheel-dir $WHEELS_DIR/mcore/ $CAUSAL_CONV1D_DIR
    pip wheel --no-deps --wheel-dir $WHEELS_DIR/mcore/ $MLM_DIR
  else
    pip install --no-cache-dir $WHEELS_DIR/mcore/*.whl
    pip install --no-cache-dir -e $MLM_DIR
  fi
}

te() {
  local mode="$1"
  TE_DIR="/opt/TransformerEngine"

  if [ ! -d "$TE_DIR/.git" ]; then
    rm -rf "$TE_DIR" &&
      cd $(dirname "$TE_DIR") &&
      git clone ${TE_REPO}
  fi &&
    pushd $TE_DIR &&
    git checkout -f $TE_TAG &&
    popd

  if [[ "$mode" == "build" ]]; then
    cd $TE_DIR && git submodule init && git submodule update &&
      pip wheel --wheel-dir $WHEELS_DIR/te/ $TE_DIR
  else
    pip install --no-cache-dir $WHEELS_DIR/te/*.whl
  fi
}

apex() {
  local mode="$1"
  APEX_DIR="/opt/Apex"

  if [ ! -d "$APEX_DIR/.git" ]; then
    rm -rf "$APEX_DIR" &&
      cd $(dirname "$APEX_DIR") &&
      git clone ${APEX_REPO}
  fi &&
    pushd $APEX_DIR &&
    git checkout -f $APEX_TAG &&
    popd

  if [[ "$mode" == "build" ]]; then
    cd $APEX_DIR && pip wheel --no-deps --no-build-isolation --wheel-dir $WHEELS_DIR/apex/ $APEX_DIR
  else
    pip install --no-cache-dir --no-build-isolation $WHEELS_DIR/apex/*.whl
  fi

}

nemo() {
  local mode="$1"
  NEMO_DIR=${NEMO_DIR:-"/opt/NeMo"}

  if [[ -n "$NEMO_TAG" ]]; then
    if [ ! -d "$NEMO_DIR/.git" ]; then
      rm -rf "$NEMO_DIR" &&
        cd $(dirname "$NEMO_DIR") &&
        git clone ${NEMO_REPO}
    fi &&
      pushd $NEMO_DIR &&
      git fetch origin '+refs/pull/*/merge:refs/remotes/pull/*/merge' &&
      git fetch origin $NEMO_TAG &&
      git checkout -f $NEMO_TAG
  fi

  ${PIP} install --no-cache-dir virtualenv &&
    virtualenv /opt/venv &&
    /opt/venv/bin/pip install --no-cache-dir --no-build-isolation \
      -r $NEMO_DIR/requirements/requirements_vllm.txt \
      -r $NEMO_DIR/requirements/requirements_deploy.txt

  DEPS=(
    "nvidia-modelopt[torch]~=0.21.0; sys_platform == 'linux'"
    "nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@34259bd3e752fef94045a9a019e4aaf62bd11ce2"
    "git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@97aad77609d2e25ed38ac5c99f0c13f93c48464e"
    "onnxscript @ git+https://github.com/microsoft/onnxscript"
    "llama-index==0.10.43"
    "unstructured==0.14.9"
    "triton==3.1.0"
  )

  echo 'Installing dependencies of nemo'
  ${PIP} install --force-re-install --no-cache-dir --extra-index-url https://pypi.nvidia.com "${DEPS[@]}"

  echo 'Installing nemo itself'
  pip install --no-cache-dir --no-build-isolation $NEMO_DIR/.[all]
}

echo 'Uninstalling stuff'
# Some of these packages are uninstalled for legacy purposes
${PIP} uninstall -y nemo_toolkit sacrebleu nemo_asr nemo_nlp nemo_tts

if [ -n "${NVIDIA_PYTORCH_VERSION}" ]; then
  echo "Installing NeMo in NVIDIA PyTorch container: ${NVIDIA_PYTORCH_VERSION}"

  echo "Will not install numba"

else
  if [ -n "${CONDA_PREFIX}" ]; then
    NUMBA_VERSION=0.57.1
    echo 'Installing numba=='${NUMBA_VERSION}
    conda install -y -c conda-forge numba==${NUMBA_VERSION}
  fi
fi

echo 'Installing nemo'
cd $CURR

if [[ "$INSTALL_OPTION" == "dev" ]]; then
  echo "Running in dev mode"
  ${PIP} install --editable ".[all]"

else
  # --------------------------
  # Argument Parsing & Validation
  # --------------------------

  # Parse command-line arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
    --library)
      LIBRARY_ARG="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
    esac
  done

  # Validate required arguments
  if [[ -z "$LIBRARY_ARG" ]]; then
    echo "Error: --library argument is required"
    exit 1
  fi

  if [[ -z "$MODE" ]]; then
    echo "Error: --mode argument is required"
    exit 1
  fi

  # Validate mode
  if [[ "$MODE" != "build" && "$MODE" != "install" ]]; then
    echo "Error: Invalid mode. Must be 'build' or 'install'"
    exit 1
  fi

  # Process library argument
  declare -a LIBRARIES
  if [[ "$LIBRARY_ARG" == "all" ]]; then
    LIBRARIES=("${ALL_LIBRARIES[@]}")
  else
    IFS=',' read -ra TEMP_ARRAY <<<"$LIBRARY_ARG"
    for lib in "${TEMP_ARRAY[@]}"; do
      trimmed_lib=$(echo "$lib" | xargs)
      if [[ -n "$trimmed_lib" ]]; then
        LIBRARIES+=("$trimmed_lib")
      fi
    done
  fi

  # Validate libraries array
  if [[ ${#LIBRARIES[@]} -eq 0 ]]; then
    echo "Error: No valid libraries specified"
    exit 1
  fi

  # Validate each library is supported
  for lib in "${LIBRARIES[@]}"; do
    if [[ ! " ${ALL_LIBRARIES[@]} " =~ " ${lib} " ]]; then
      echo "Error: Unsupported library '$lib'"
      exit 1
    fi
  done

  # --------------------------
  # Execution Logic
  # --------------------------

  # Run operations for each library
  for library in "${LIBRARIES[@]}"; do
    echo "Processing $library ($MODE)..."
    "$library" "$MODE"

    # Check if function succeeded
    if [[ $? -ne 0 ]]; then
      echo "Error: Operation failed for $library"
      exit 1
    fi
  done

  echo "All operations completed successfully"
  exit 0

fi

echo 'All done!'
