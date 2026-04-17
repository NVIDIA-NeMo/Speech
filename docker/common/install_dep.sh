#!/usr/bin/env bash
set -exou pipefail

# List of all supported libraries (update this list when adding new libraries)
# This also defines the order in which they will be installed by --libraries "all"

ALL_LIBRARIES=(
  "trtllm"
  "vllm"
  "extra"
)

export INSTALL_OPTION=${1:-dev}
export HEAVY_DEPS=${HEAVY_DEPS:-false}
export INSTALL_DIR=${INSTALL_DIR:-"/opt"}
export CURR=$(pwd)
export WHEELS_DIR=${WHEELS_DIR:-"$INSTALL_DIR/wheels"}
export PIP=pip
export TRTLLM_REPO=${TRTLLM_REPO:-$(cat "$CURR/requirements/manifest.json" | jq -r '."vcs-dependencies"."trt-llm".repo')}
export TRTLLM_TAG=${TRTLLM_TAG:-$(cat "$CURR/requirements/manifest.json" | jq -r '."vcs-dependencies"."trt-llm".ref')}
export TRTLLM_DIR="$INSTALL_DIR/TensorRT-LLM"
export NVIDIA_PYTORCH_VERSION=${NVIDIA_PYTORCH_VERSION:-""}
export CONDA_PREFIX=${CONDA_PREFIX:-""}

trt() {
  local mode="$1"
  local WHEELS_DIR=$WHEELS_DIR/trt/
  mkdir -p $WHEELS_DIR

  # Skip TRT installation on macOS ARM
  if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    echo "Skipping TRT installation on macOS ARM"
    return
  fi

  if [ "$(id -u)" -ne 0 ]; then
    if ! command -v sudo &>/dev/null; then
      echo "Not running as root and sudo is not available, skipping TRT installation"
      return
    fi
  fi

  if [ "$(id -u)" -eq 0 ]; then
    # Already root, run directly
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt-get install git-lfs
    git lfs install
    apt-get clean
  else
    # Need to gain sudo
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install
    sudo apt-get clean
  fi

  if [ ! -d "$TRTLLM_DIR/.git" ]; then
    rm -rf "$TRTLLM_DIR"
    cd $(dirname "$TRTLLM_DIR")
    git clone ${TRTLLM_REPO}
  fi

  pushd $TRTLLM_DIR
  git checkout -f $TRTLLM_TAG
  git submodule update --init --recursive
  sed -i "/torch/d" requirements.txt
  git lfs pull
  patch -p1 < $CURR/external/patches/trt_llm.patch
  popd

  if [[ "$mode" == "install" ]]; then
    if [[ "${NVIDIA_PYTORCH_VERSION}" != "" ]]; then
      cd $TRTLLM_DIR
      set +u

      bash docker/common/install_base.sh
      bash docker/common/install_cmake.sh
      bash docker/common/install_ccache.sh

      . docker/common/install_tensorrt.sh \
        --TRT_VER="10.10.0.31" \
        --CUDA_VER="12.9" \
        --CUDNN_VER="9.9.0.52-1" \
        --NCCL_VER="2.26.5-1+cuda12.9" \
        --CUBLAS_VER="12.9.0.13-1" \
        --NVRTC_VER="12.9.41-1"
      set -u
    fi
  fi
}

trtllm() {
  local mode="$1"
  local WHEELS_DIR=$WHEELS_DIR/trtllm/
  mkdir -p $WHEELS_DIR

  # Skip TRT installation on macOS ARM
  if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    echo "Skipping TRT installation on macOS ARM"
    return
  fi

  if [ "$(id -u)" -ne 0 ]; then
    if ! command -v sudo &>/dev/null; then
      echo "Not running as root and sudo is not available, skipping TRT installation"
      return
    fi
  fi

  if [ "$(id -u)" -eq 0 ]; then
    # Already root, run directly
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt-get install git-lfs
    git lfs install
    apt-get clean
  else
    # Need to gain sudo
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install
    sudo apt-get clean
  fi

  if [ ! -d "$TRTLLM_DIR/.git" ]; then
    rm -rf "$TRTLLM_DIR"
    cd $(dirname "$TRTLLM_DIR")
    git clone ${TRTLLM_REPO}
  fi
  pushd $TRTLLM_DIR
  git checkout -f $TRTLLM_TAG
  git submodule update --init --recursive
  sed -i "/torch/d" requirements.txt
  git lfs pull
  patch -p1 < $CURR/external/patches/trt_llm.patch
  popd

  build() {
    if [[ "${NVIDIA_PYTORCH_VERSION}" != "" ]]; then
      # CONDA_PREFIX causes an error in trt-llm's build script
      unset CONDA_PREFIX
      cd $TRTLLM_DIR
      TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" python3 ./scripts/build_wheel.py --job_count $(nproc) --clean --trt_root /usr/local/tensorrt --dist_dir $WHEELS_DIR --python_bindings --benchmarks
    fi
  }

  if [[ "$mode" == "build" ]]; then
    build
  else
    if [ -d "$WHEELS_DIR" ] && [ -z "$(ls -A "$WHEELS_DIR")" ]; then
      build
    fi

    pip install --no-cache-dir $WHEELS_DIR/tensorrt_llm*.whl --extra-index-url https://pypi.nvidia.com || true
  fi
}

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
# Some packages in pytorch:26.03-py3 are debian-installed without a RECORD file,
# so pip cannot uninstall them during dependency resolution. Pre-install them
# with --ignore-installed so pip places pip-managed copies in the higher-priority
# /usr/local site-packages directory, shadowing the debian versions.
${PIP} install --no-cache-dir --ignore-installed wheel PyYAML
${PIP} install -U --no-cache-dir "setuptools==76.0.0" pybind11 ${PIP}

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
    # "trt" is a valid option but not in ALL_LIBRARIES
    # It does not get installed at the same time as the rest
    if [[ "$lib" == "trt" ]]; then
      continue
    fi

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
