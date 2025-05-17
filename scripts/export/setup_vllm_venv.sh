# This script sets up a Python virtual environment
# with all the requirements for running vLLM.
set -ex

VENV_DIR="${1:-/opt/venv}"

echo "Creating virtual environment in ${VENV_DIR}..."

pip install virtualenv

virtualenv ${VENV_DIR}

${VENV_DIR}/bin/pip install \
    -r /opt/NeMo/requirements/requirements_vllm.txt \
    -r /opt/NeMo/requirements/requirements_deploy.txt
