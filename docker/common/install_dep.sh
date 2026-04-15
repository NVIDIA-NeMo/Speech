#!/usr/bin/env bash
set -exou pipefail

# List of all supported libraries (update this list when adding new libraries)
# This also defines the order in which they will be installed by --libraries "all"

ALL_LIBRARIES=(
  "vllm"
  "extra"
)

export INSTALL_OPTION=${1:-dev}
export HEAVY_DEPS=${HEAVY_DEPS:-false}
export INSTALL_DIR=${INSTALL_DIR:-"/opt"}
export CURR=$(pwd)
export WHEELS_DIR=${WHEELS_DIR:-"$INSTALL_DIR/wheels"}
export PIP=pip
export NVIDIA_PYTORCH_VERSION=${NVIDIA_PYTORCH_VERSION:-""}
export CONDA_PREFIX=${CONDA_PREFIX:-""}

vllm() {
  local mode="$1"

  local WHEELS_DIR=$WHEELS_DIR/vllm/
  mkdir -p $WHEELS_DIR

  VLLM_DIR="$INSTALL_DIR/vllm"

  build() {
    if [[ "${NVIDIA_PYTORCH_VERSION}" != "" ]]; then
      ${PIP} install --no-cache-dir virtualenv
      virtualenv $INSTALL_DIR/venv
      $INSTALL_DIR/venv/bin/pip install --no-cache-dir setuptools coverage
      $INSTALL_DIR/venv/bin/pip wheel --no-cache-dir --no-build-isolation \
        --wheel-dir $WHEELS_DIR/ \
        -r $CURR/requirements/requirements_vllm.txt
    fi
  }

  if [[ "$mode" == "build" ]]; then
    build
  else
    if [ -d "$WHEELS_DIR" ] && [ -z "$(ls -A "$WHEELS_DIR")" ]; then
      build
    fi

    ${PIP} install --no-cache-dir virtualenv
    virtualenv $INSTALL_DIR/venv
    $INSTALL_DIR/venv/bin/pip install --no-cache-dir coverage
    $INSTALL_DIR/venv/bin/pip install --no-cache-dir --no-build-isolation $WHEELS_DIR/*.whl || true
  fi

}

extra() {
  local mode="$1"
  DEPS=(
    "llama-index==0.10.43"                                                                     # incompatible with nvidia-pytriton
    "nemo_run"
    "nvidia-modelopt==0.37.0"                                                                  # We want a specific version of nvidia-modelopt
  )
  if [[ "${NVIDIA_PYTORCH_VERSION}" != "" ]]; then
    DEPS+=(
      "git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@b6eb61dbf9fe272b1a943b1b0d9efdde99df0737 ; platform_machine == 'x86_64'" # Compiling NvRX requires CUDA
    )
  fi

  if [[ "$mode" == "install" ]]; then
    pip install --force-reinstall --no-deps --no-cache-dir "${DEPS[@]}"
    pip install --no-cache-dir "${DEPS[@]}"
    # needs no-deps to avoid installing triton on top of pytorch-triton.
    pip install --no-deps --no-cache-dir "liger-kernel==0.5.8; (platform_machine == 'x86_64' and platform_system != 'Darwin')"
    pip install --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@87a86aba72cfd2f0d8abecaf81c13c4528ea07d8; (platform_machine == 'x86_64' and platform_system != 'Darwin')"
  fi
}

echo 'Uninstalling stuff'
# Some of these packages are uninstalled for legacy purposes
${PIP} uninstall -y nemo_toolkit sacrebleu nemo_asr nemo_nlp nemo_tts

echo 'Upgrading tools'
${PIP} install -U --no-cache-dir "setuptools==76.0.0" pybind11 wheel ${PIP}

if [ "${NVIDIA_PYTORCH_VERSION}" != "" ]; then
  echo "Installing NeMo in NVIDIA PyTorch container: ${NVIDIA_PYTORCH_VERSION}"
  echo "Will not install numba"

else
  if [ "${CONDA_PREFIX}" != "" ]; then
    echo 'Installing numba'
    conda install -y -c conda-forge numba
  else
    pip install --no-cache-dir --no-deps torch cython
  fi
fi

echo 'Installing nemo dependencies'
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
