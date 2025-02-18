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
INSTALL_DIR=${INSTALL_DIR:-'/opt'}

PIP=pip
${PIP} install -U ${PIP}

mcore() {
  local mode="$1"

  export CAUSAL_CONV1D_FORCE_BUILD=TRUE
  export CAUSAL_CONV_TAG=v1.2.2.post1
  CAUSAL_CONV1D_DIR="$INSTALL_DIR/causal-conv1d" &&
    if [ ! -d "$CAUSAL_CONV1D_DIR/.git" ]; then
      rm -rf "$CAUSAL_CONV1D_DIR" &&
        cd $(dirname "$CAUSAL_CONV1D_DIR") &&
        git clone https://github.com/Dao-AILab/$(basename $CAUSAL_CONV1D_DIR).git
    fi &&
    pushd $CAUSAL_CONV1D_DIR &&
    git checkout -f $CAUSAL_CONV_TAG &&
    popd

  export MAMBA_FORCE_BUILD=TRUE
  export MAMBA_TAG=v2.2.0
  MAMBA_DIR="$INSTALL_DIR/mamba" &&
    if [ ! -d "$MAMBA_DIR/.git" ]; then
      rm -rf "$MAMBA_DIR" &&
        cd $(dirname "$MAMBA_DIR") &&
        git clone https://github.com/state-spaces/$(basename $MAMBA_DIR).git &&
        pushd $(basename $MAMBA_DIR) &&
        sed -i "/triton/d" setup.py &&
        popd
    fi &&
    pushd $MAMBA_DIR &&
    git checkout -f $MAMBA_TAG &&
    popd

  export MLM_REPO=${MLM_REPO:-"https://github.com/NVIDIA/Megatron-LM.git"}
  export MLM_TAG=${MLM_TAG:-"main"}
  MLM_DIR="$INSTALL_DIR/Megatron-LM" &&
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
    pip install --no-cache-dir $WHEELS_DIR/mcore/*.whl "nvidia-pytriton ; platform_machine == 'x86_64'"
    pip install --no-cache-dir -e $MLM_DIR
  fi
}

te() {
  local mode="$1"
  TE_DIR="$INSTALL_DIR/TransformerEngine"

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
  APEX_DIR="$INSTALL_DIR/Apex"

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
  NEMO_DIR=${NEMO_DIR:-"$INSTALL_DIR/NeMo"}

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

  PLATFORM_MACHINE=$(python -c "import platform; print(platform.machine())")
  if [[ "$PLATFORM_MACHINE" == "x86_64" ]]; then
    ${PIP} install --no-cache-dir virtualenv &&
      virtualenv $INSTALL_DIR/venv &&
      $INSTALL_DIR/venv/bin/pip install --no-cache-dir setuptools &&
      $INSTALL_DIR/venv/bin/pip install --no-cache-dir --no-build-isolation \
        -r $NEMO_DIR/requirements/requirements_vllm.txt \
        -r $NEMO_DIR/requirements/requirements_deploy.txt
  else
    echo "Skipping VLLM install for non x86_64 machine."
  fi

  DEPS=(
    "nvidia-modelopt[torch]~=0.21.0; sys_platform == 'linux'"
    "nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@34259bd3e752fef94045a9a019e4aaf62bd11ce2"
    "onnxscript @ git+https://github.com/microsoft/onnxscript"
    "llama-index==0.10.43"
    "unstructured==0.14.9"
  )

  if [ -n "${NVIDIA_PYTORCH_VERSION}" ]; then
    echo "Installing NVIDIA Resiliency in NVIDIA PyTorch container: ${NVIDIA_PYTORCH_VERSION}"
    pip install --no-cache-dir "git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@97aad77609d2e25ed38ac5c99f0c13f93c48464e ; platform_machine == 'x86_64'"
  fi

  echo 'Installing dependencies of nemo'
  ${PIP} install --no-cache-dir --extra-index-url https://pypi.nvidia.com "${DEPS[@]}"

  echo 'Installing nemo itself'
  pip install --no-cache-dir --no-build-isolation $NEMO_DIR/.[all]
}

echo 'Uninstalling stuff'
# Some of these packages are uninstalled for legacy purposes
${PIP} uninstall -y nemo_toolkit sacrebleu nemo_asr nemo_nlp nemo_tts

${PIP} install setuptools pybind11 wheel

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
