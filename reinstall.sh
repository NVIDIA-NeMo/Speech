#!/usr/bin/env bash
set -ex

# List of all supported libraries (update this list when adding new libraries)
# This also defines the order in which they will be installed by --libraries "all"
ALL_LIBRARIES=(
  "trtllm"
  "mcore"
  "nemo"
  "vllm"
)

export INSTALL_OPTION=${1:-dev}
export HEAVY_DEPS=${HEAVY_DEPS:-false}
export CURR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export INSTALL_DIR=${INSTALL_DIR:-"/opt"}
export WHEELS_DIR=${WHEELS_DIR:-"$INSTALL_DIR/wheels"}
export PIP=pip
export TRTLLM_REPO=${TRTLLM_REPO:-$(cat "$CURR/requirements/manifest.json" | jq -r '."vcs-dependencies"."trt_llm".repo')}
export TRTLLM_TAG=${TRTLLM_TAG:-$(cat "$CURR/requirements/manifest.json" | jq -r '."vcs-dependencies"."trt_llm".ref')}
export TRTLLM_DIR="$INSTALL_DIR/TensorRT-LLM"

trt() {
  local mode="$1"
  local WHEELS_DIR=$WHEELS_DIR/trt/
  mkdir -p $WHEELS_DIR

  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash &&
    apt-get install git-lfs &&
    git lfs install &&
    apt-get clean

  if [ ! -d "$TRTLLM_DIR/.git" ]; then
    rm -rf "$TRTLLM_DIR" &&
      cd $(dirname "$TRTLLM_DIR") &&
      git clone ${TRTLLM_REPO}
  fi &&
    pushd $TRTLLM_DIR &&
    git checkout -f $TRTLLM_TAG &&
    git lfs pull &&
    popd

  if [[ "$mode" == "install" ]]; then
    if [[ -n "${NVIDIA_PYTORCH_VERSION}" ]]; then
      cd $TRTLLM_DIR &&
        . docker/common/install_tensorrt.sh \
          --TRT_VER="10.8.0.43" \
          --CUDA_VER="12.8" \
          --CUDNN_VER="9.7.0.66-1" \
          --NCCL_VER="2.25.1-1+cuda12.8" \
          --CUBLAS_VER="12.8.3.14-1"
    fi
  fi
}

trtllm() {
  local mode="$1"
  local WHEELS_DIR=$WHEELS_DIR/trtllm/
  mkdir -p $WHEELS_DIR

  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash &&
    apt-get install git-lfs &&
    git lfs install &&
    apt-get clean

  if [ ! -d "$TRTLLM_DIR/.git" ]; then
    rm -rf "$TRTLLM_DIR" &&
      cd $(dirname "$TRTLLM_DIR") &&
      git clone ${TRTLLM_REPO}
  fi &&
    pushd $TRTLLM_DIR &&
    git checkout -f $TRTLLM_TAG &&
    git lfs pull &&
    popd

  build() {
    if [[ -n "${NVIDIA_PYTORCH_VERSION}" ]]; then
      cd $TRTLLM_DIR &&
        python3 ./scripts/build_wheel.py --job_count $(nproc) --trt_root /usr/local/tensorrt --dist_dir $WHEELS_DIR/trtllm/ --python_bindings --benchmarks
    fi
  }

  if [[ "$mode" == "build" ]]; then
    build
  else
    if [ -d "$WHEELS_DIR" ] && [ -z "$(ls -A "$WHEELS_DIR")" ]; then
      build
    fi

    pip install --no-cache-dir $WHEELS_DIR/trtllm/tensorrt_llm*.whl --extra-index-url https://pypi.nvidia.com || true
  fi
}

te() {
  local mode="$1"
  local WHEELS_DIR=$WHEELS_DIR/te/
  mkdir -p $WHEELS_DIR

  TE_REPO=${TE_REPO:-$(cat "$CURR/requirements/manifest.json" | jq -r '."vcs-dependencies"."transformer_engine".repo')}
  TE_TAG=${TE_TAG:-$(cat "$CURR/requirements/manifest.json" | jq -r '."vcs-dependencies"."transformer_engine".ref')}
  TE_DIR="$INSTALL_DIR/TransformerEngine"
  if [ ! -d "$TE_DIR/.git" ]; then
    rm -rf "$TE_DIR" &&
      cd $(dirname "$TE_DIR") &&
      git clone ${TE_REPO}
  fi &&
    pushd $TE_DIR &&
    git checkout -f $TE_TAG &&
    patch -p1 </tmp/NeMo/external/patches/nemo_2.3.0_te.patch &&
    popd

  build() {
    if [[ -n "${NVIDIA_PYTORCH_VERSION}" ]]; then
      cd $TE_DIR && git submodule init && git submodule update &&
        pip wheel --wheel-dir $WHEELS_DIR/te/ $TE_DIR
    fi
  }

  if [[ "$mode" == "build" ]]; then
    build
  else
    if [ -d "$WHEELS_DIR" ] && [ -z "$(ls -A "$WHEELS_DIR")" ]; then
      build
    fi

    pip install --no-cache-dir $WHEELS_DIR/te/*.whl || true
  fi
}

mcore() {
  local mode="$1"

  local WHEELS_DIR=$WHEELS_DIR/mcore/
  mkdir -p $WHEELS_DIR

  export CAUSAL_CONV1D_FORCE_BUILD=TRUE
  export CAUSAL_CONV_TAG=v1.2.2.post1
  CAUSAL_CONV1D_DIR="$INSTALL_DIR/causal-conv1d" &&
    if [ ! -d "$CAUSAL_CONV1D_DIR/.git" ]; then
      rm -rf "$CAUSAL_CONV1D_DIR" &&
        mkdir -p $(dirname "$CAUSAL_CONV1D_DIR") &&
        cd $(dirname "$CAUSAL_CONV1D_DIR") &&
        git clone https://github.com/Dao-AILab/$(basename $CAUSAL_CONV1D_DIR).git
    fi &&
    pushd $CAUSAL_CONV1D_DIR &&
    git checkout -f $CAUSAL_CONV_TAG &&
    popd

  export MAMBA_FORCE_BUILD=TRUE
  export MAMBA_TAG=2e16fc3062cdcd4ebef27a9aa4442676e1c7edf4
  MAMBA_DIR="$INSTALL_DIR/mamba" &&
    if [ ! -d "$MAMBA_DIR/.git" ]; then
      rm -rf "$MAMBA_DIR" &&
        cd $(dirname "$MAMBA_DIR") &&
        git clone https://github.com/state-spaces/$(basename $MAMBA_DIR).git
    fi &&
    pushd $MAMBA_DIR &&
    git checkout -f $MAMBA_TAG &&
    perl -ni -e 'print unless /triton/' setup.py &&
    perl -ni -e 'print unless /triton/' pyproject.toml &&
    popd

  MLM_REPO=${MLM_REPO:-$(cat "$CURR/requirements/manifest.json" | jq -r '."vcs-dependencies"."megatron-lm".repo')}
  MLM_TAG=${MLM_TAG:-$(cat "$CURR/requirements/manifest.json" | jq -r '."vcs-dependencies"."megatron-lm".ref')}
  MLM_DIR="$INSTALL_DIR/Megatron-LM" &&
    if [ ! -d "$MLM_DIR/.git" ]; then
      rm -rf "$MLM_DIR" &&
        mkdir -p $(dirname "$MLM_DIR") &&
        cd $(dirname "$MLM_DIR") &&
        git clone ${MLM_REPO}
    fi &&
    pushd $MLM_DIR &&
    git checkout -f $MLM_TAG &&
    perl -ni -e 'print unless /triton==3.1.0/' requirements/pytorch_24.10/requirements.txt &&
    perl -ni -e 'print unless /nvidia-resiliency-ext/' requirements/pytorch_24.10/requirements.txt &&
    popd

  build() {
    if [[ -n "${NVIDIA_PYTORCH_VERSION}" ]]; then
      pip wheel --no-deps --no-cache-dir --wheel-dir $WHEELS_DIR $MAMBA_DIR
      pip wheel --no-deps --no-cache-dir --wheel-dir $WHEELS_DIR $CAUSAL_CONV1D_DIR
    fi

    pip wheel --no-deps --wheel-dir $WHEELS_DIR $MLM_DIR
  }

  if [[ "$mode" == "build" ]]; then
    build
  else
    if [ -d "$WHEELS_DIR" ] && [ -z "$(ls -A "$WHEELS_DIR")" ]; then
      build
    fi

    pip install --no-cache-dir $WHEELS_DIR/*.whl "nvidia-pytriton ; platform_machine == 'x86_64'" || true
    pip install --no-cache-dir -e $MLM_DIR
  fi
}

vllm() {
  local mode="$1"

  local WHEELS_DIR=$WHEELS_DIR/vllm/
  mkdir -p $WHEELS_DIR

  VLLM_DIR="$INSTALL_DIR/vllm"

  build() {
    if [[ -n "${NVIDIA_PYTORCH_VERSION}" ]]; then
      ${PIP} install --no-cache-dir virtualenv &&
        virtualenv $INSTALL_DIR/venv &&
        $INSTALL_DIR/venv/bin/pip install --no-cache-dir setuptools coverage &&
        $INSTALL_DIR/venv/bin/pip wheel --no-cache-dir --no-build-isolation \
          --wheel-dir $WHEELS_DIR/ \
          -r $NEMO_DIR/requirements/requirements_vllm.txt \
          -r $NEMO_DIR/requirements/requirements_deploy.txt
    fi
  }

  if [[ "$mode" == "build" ]]; then
    build
  else
    if [ -d "$WHEELS_DIR" ] && [ -z "$(ls -A "$WHEELS_DIR")" ]; then
      build
    fi

    ${PIP} install --no-cache-dir virtualenv &&
      virtualenv $INSTALL_DIR/venv &&
      $INSTALL_DIR/venv/bin/pip install --no-cache-dir coverage &&
      $INSTALL_DIR/venv/bin/pip install --no-cache-dir --no-build-isolation $WHEELS_DIR/*.whl || true
  fi

}

nemo() {
  local mode="$1"

  if [[ "$mode" == "build" ]]; then
    echo "No build supported for Nemo, directly install."
    return
  fi

  NEMO_DIR=${NEMO_DIR:-"$INSTALL_DIR/NeMo"}
  if [[ -n "$NEMO_TAG" ]]; then
    if [ ! -d "$NEMO_DIR/.git" ]; then
      rm -rf "$NEMO_DIR" &&
        mkdir -p $(dirname "$NEMO_DIR") &&
        cd $(dirname "$NEMO_DIR") &&
        git clone ${NEMO_REPO}
    fi &&
      pushd $NEMO_DIR &&
      git fetch origin '+refs/pull/*/merge:refs/remotes/pull/*/merge' &&
      git fetch origin $NEMO_TAG &&
      git checkout -f $NEMO_TAG
  else
    NEMO_DIR=$CURR
  fi

  DEPS=(
    "llama-index==0.10.43"                                                                     # incompatible with nvidia-pytriton
    "ctc_segmentation==1.7.1 ; (platform_machine == 'x86_64' and platform_system != 'Darwin')" # requires numpy<2.0.0 to be installed before
    "nemo_run"                                                                                 # Not compatible in Python 3.12
    "nvidia-modelopt[torch]==0.27.1 ; platform_system != 'Darwin'"                             # We want a specific version of nvidia-modelopt
  )

  echo 'Installing dependencies of nemo'
  pip install --force-reinstall --no-deps --no-cache-dir "${DEPS[@]}"
  pip install --no-cache-dir "${DEPS[@]}"
  # needs no-deps to avoid installing triton on top of pytorch-triton.
  pip install --no-deps --no-cache-dir "liger-kernel==0.5.4; (platform_machine == 'x86_64' and platform_system != 'Darwin')"
  pip install --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@87a86aba72cfd2f0d8abecaf81c13c4528ea07d8; (platform_machine == 'x86_64' and platform_system != 'Darwin')"
  echo 'Installing nemo itself'
  pip install --no-cache-dir -e $NEMO_DIR/.[all]
}

echo 'Uninstalling stuff'
# Some of these packages are uninstalled for legacy purposes
${PIP} uninstall -y nemo_toolkit sacrebleu nemo_asr nemo_nlp nemo_tts

echo 'Upgrading tools'
${PIP} install -U --no-cache-dir "setuptools==76.0.0" pybind11 wheel ${PIP}

if [ -n "${NVIDIA_PYTORCH_VERSION}" ]; then
  echo "Installing NeMo in NVIDIA PyTorch container: ${NVIDIA_PYTORCH_VERSION}"
  echo "Will not install numba"

else
  if [ -n "${CONDA_PREFIX}" ]; then
    echo 'Installing numba'
    conda install -y -c conda-forge numba
  else
    pip install --no-cache-dir --no-deps torch cython
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
