#!/usr/bin/env bash
# Install OS dependencies and set up a Python environment with uv.
#
# Usage:
#   bash install.sh
#
# Prerequisites:
#   - Linux (apt-based)
#   - A CUDA 13.0-compatible GPU + driver for the default PyTorch wheels.
#     To use a different CUDA version or CPU-only, edit [tool.uv.sources]
#     in pyproject.toml before running this script.

set -euo pipefail

# --- 0. Pre-flight check --------------------------------------------------
# uv manages its own virtualenv. Running inside a conda env causes toolchain
# mismatches (conda's gcc + system Python headers) that break C extensions
# like cdifflib. Bail out early if conda is active.
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    echo "Error: conda env '${CONDA_DEFAULT_ENV}' is active."
    echo "Please run 'conda deactivate' first — uv manages its own venv."
    exit 1
fi

# --- 1. OS-level dependencies ---------------------------------------------
# - npm + nodejs: needed for the browser client in client/
# - build-essential: C/C++ compilers for packages that build from source
# - python3-dev: Python.h headers required by cdifflib and other C extensions
#   (nemo-toolkit[tts] → cdifflib needs this to compile)
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y npm nodejs build-essential python3-dev
else
    echo "Warning: apt-get not found. Install npm, nodejs, a C/C++ toolchain, and Python.h headers manually."
fi

# --- 2. Install uv --------------------------------------------------------
# uv is a fast Python package manager (replaces pip + virtualenv + pyenv).
# It's a single static binary; this installer drops it in ~/.local/bin.
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv visible in this shell session
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- 3. Create the virtual environment and install Python dependencies ----
# `uv sync` reads pyproject.toml (and uv.lock if present) and creates a
# .venv/ in the current directory with everything installed.
uv sync

echo
echo "✓ Install complete."
echo "  Activate the env with:   source .venv/bin/activate"
echo "  Or run any command with: uv run <command>"
