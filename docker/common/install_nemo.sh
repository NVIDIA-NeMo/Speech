#!/usr/bin/env bash
set -exou pipefail

export INSTALL_DIR=${INSTALL_DIR:-"/opt"}
export CURR=$(pwd)
export NVIDIA_PYTORCH_VERSION=${NVIDIA_PYTORCH_VERSION:-""}
export PIP=pip

echo 'Installing nemo'
cd $CURR

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
  --module)
    MODULE_ARG="$2"
    shift 2
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

NEMO_DIR=${NEMO_DIR:-"$INSTALL_DIR/NeMo"}
NEMO_REPO=${NEMO_REPO:-"https://github.com/NVIDIA/NeMo"}
NEMO_TAG=${NEMO_TAG:-""}
if [[ -n "$NEMO_TAG" ]]; then
  if [ ! -d "$NEMO_DIR/.git" ]; then
    rm -rf "$NEMO_DIR"
    mkdir -p $(dirname "$NEMO_DIR")
    cd $(dirname "$NEMO_DIR")
    git clone ${NEMO_REPO}
  fi
  pushd $NEMO_DIR
  git fetch origin '+refs/pull/*/merge:refs/remotes/pull/*/merge'
  git fetch origin $NEMO_TAG
  git checkout -f $NEMO_TAG
else
  NEMO_DIR=$CURR
fi

DEPS=(
  "llama-index==0.10.43"                                                                     # incompatible with nvidia-pytriton
  "ctc_segmentation==1.7.1 ; (platform_machine == 'x86_64' and platform_system != 'Darwin')" # requires numpy<2.0.0 to be installed before
  "nemo_run"
  "nvidia-modelopt[torch]==0.27.1 ; platform_system != 'Darwin'"                             # We want a specific version of nvidia-modelopt
)

if [[ "${NVIDIA_PYTORCH_VERSION}" != "" ]]; then
  DEPS+=(
    "git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@b6eb61dbf9fe272b1a943b1b0d9efdde99df0737 ; platform_machine == 'x86_64'" # Compiling NvRX requires CUDA
  )
fi

${PIP} uninstall -y nemo_toolkit sacrebleu nemo_asr nemo_nlp nemo_tts
echo 'Installing dependencies of nemo'
pip install --force-reinstall --no-deps --no-cache-dir "${DEPS[@]}"
pip install --no-cache-dir "${DEPS[@]}"
# needs no-deps to avoid installing triton on top of pytorch-triton.
pip install --no-deps --no-cache-dir "liger-kernel==0.5.8; (platform_machine == 'x86_64' and platform_system != 'Darwin')"
pip install --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@87a86aba72cfd2f0d8abecaf81c13c4528ea07d8; (platform_machine == 'x86_64' and platform_system != 'Darwin')"

echo 'Installing nemo itself'
pip install --no-cache-dir -e $NEMO_DIR/.["$MODULE_ARG",test]

if [[ "${NVIDIA_PYTORCH_VERSION}" != "" ]]; then
  patch \
    /usr/local/lib/python3.12/dist-packages/torch/accelerator/__init__.py \
    /$CURR/external/patches/torch_accelerator_144567_fix.patch
fi
