[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![CodeQL](https://github.com/NVIDIA-NeMo/Speech/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/NVIDIA-NeMo/Speech/actions/workflows/codeql.yml)
[![NeMo Speech license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA-NeMo/Speech/blob/main/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# **NVIDIA NeMo Speech**
Checkout our [HuggingFace🤗 collection](https://huggingface.co/collections/nvidia/nemotron-speech) for the latest open
weight checkpoints and demos!

## Updates

> NeMo Speech 3.0 is now available as [release v3.0.0](https://github.com/NVIDIA-NeMo/Speech/releases/tag/v3.0.0)
> and in the [26.07.00 NeMo Speech NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-speech?version=26.07.00).
> The final NeMo release before the repository split was
> [v2.7.3](https://github.com/NVIDIA-NeMo/Speech/releases/tag/v2.7.3), available in the
> [26.04 NeMo NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=26.04).

- 2026-07: [MagpieTTS v2607](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) has been released with support
    for 3 new languages (Ar, Ko, Pt) + 9 existing languages (En, Es, De, Fr, Vi, It, Zh, Hi, Ja). Try out
    [the demo](https://huggingface.co/spaces/nvidia/magpie_tts_multilingual_demo)!
- 2026-06: [Nemotron-3.5-ASR-Streaming-0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) has been released with 40 languages supported, controllable latency 80ms-1s, and 240-2400 1xH100 concurrent streams. Built on cache-aware Fastconformer architecture.
- 2026-04: [Parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) has been released with high-quality offline and streaming (with a minimum latency of 160ms) inference in one model for English language with punctuation and capitalization support. 
- 2026-03: [Nemotron 3 VoiceChat](https://build.nvidia.com/nvidia/nemotron-voicechat/modelcard) is now released in Early Access. Built on the Nemotron Nano v2 LLM backbone with Nemotron speech and TTS decoder, VoiceChat delivers full-duplex, natural, interruptible conversations with low latency. Try out [the demo](https://build.nvidia.com/nvidia/nemotron-voicechat) and apply for [early access](https://developer.nvidia.com/nemotron-voicechat-early-access).
- 2026-03: [Nemotron-Speech-Streaming v2603](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) has been
    updated. It has been trained on a larger and more diverse corpus, resulting in lower WER across all latency modes.
    Try out [the demo](https://huggingface.co/spaces/nvidia/nemotron-speech-streaming-en-0.6b) and check out
    [the NIM](https://build.nvidia.com/nvidia/nemotron-asr-streaming).
- 2026-03: [MagpieTTS v2602](https://huggingface.co/nvidia/magpie_tts_multilingual_357m/tree/v2602) has been released with support
    for 9 languages (En, Es, De, Fr, Vi, It, Zh, Hi, Ja).
- 2026-01: Nemotron-Speech-Streaming was released: One checkpoint that enables users to pick their optimal point
    on the latency-accuracy Pareto curve!
- 2026-01: [MagpieTTS v2512](https://huggingface.co/nvidia/magpie_tts_multilingual_357m/tree/v2512) was released.
- 2026: This repo has pivoted to focus on audio, speech, and multimodal LLMs. For the final pre-split NeMo release with
    support for additional modalities, see [v2.7.3](https://github.com/NVIDIA-NeMo/Speech/releases/tag/v2.7.3).
- 2025-08: [Parakeet V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and
    [Canary V2](https://huggingface.co/nvidia/canary-1b-v2) have been released with speech recognition and translation
    support for 25 European languages.
- 2025-06: [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) has been released with record-setting
    5.63% WER on English Open ASR Leaderboard.

## Introduction

NVIDIA NeMo Speech is built for researchers and PyTorch developers working on Speech models including Automatic Speech
Recognition (ASR), Text to Speech (TTS), and Speech LLMs. It is designed to help you efficiently create, customize, and
deploy new AI models by leveraging existing code and pre-trained model checkpoints.

For technical documentation, please see the
[NeMo Speech Developer Documentation](https://docs.nvidia.com/nemo/speech/nightly/).

## Requirements

NeMo Speech works with the **Python, PyTorch, and CUDA versions of your choosing**:

- Python 3.12 or above
- PyTorch 2.7 or above (CPU, CUDA, etc. — your choice)
- NVIDIA GPU + CUDA (required for training; recommended for inference)

If you already have a Python/PyTorch/CUDA stack that satisfies those minimums, NeMo Speech installs on top of it **without replacing it**, so your existing PyTorch build is kept (see the install options below). The versions pinned in `uv.lock` and shipped in the official container — Python 3.13, PyTorch 2.11 with CUDA 12.9 or PyTorch 2.12 with CUDA 13.2 — are simply the combinations we actively test and support. They make setup turnkey and reproducible, but they are **not** a hard requirement.

As of [Pytorch 2.6](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true),
`torch.load` defaults to using `weights_only=True`. Some model checkpoints may require using `weights_only=False`.
In this case, you can set the env var `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before running code that uses `torch.load`.
However, this should only be done with trusted files. Loading files from untrusted sources with more than weights only
can have the risk of arbitrary code execution.

## Developer Documentation

| Version | Description |
| ------- | ----------- |
| 3.0.0 (latest release) | [NeMo Speech 3.0.0 documentation](https://docs.nvidia.com/nemo/speech/3.0.0/) |
| Nightly | [Documentation for the latest `main` branch](https://docs.nvidia.com/nemo/speech/nightly/) |

## Install NeMo Speech

The recommended way to install NeMo Speech is from source with [uv](https://docs.astral.sh/uv/), which reproduces our actively-tested stack from the committed `uv.lock`. If you need different Python/PyTorch/CUDA versions, NeMo Speech also installs over your existing environment via pip — see the [pip fallback](#from-pypi-with-pip-fallback--bring-your-own-versions) below.

### From source with uv (recommended)

```bash
git clone https://github.com/NVIDIA-NeMo/Speech.git
cd Speech
uv sync --extra all --extra cu13     # CUDA 13.x (recommended) — use --extra cu12 for CUDA 12.x
```

This installs our supported stack (Python 3.13, PyTorch 2.12, CUDA 13.2) into `.venv/` with NeMo Speech editable. Add `--group test` for the test suite or `--group docs` to build the docs; run tools via `uv run <cmd>` or activate with `source .venv/bin/activate`. On Linux, `cu12` and `cu13` are mutually exclusive — pass exactly one (`cu13` is the default). For the **exact** container baseline, add `--locked --python 3.13` (the path the Dockerfile and CI use).

> **SpeechLM2 / Automodel:** the Automodel backend runs **without** any compiled dependencies. It can *optionally* benefit from dedicated accelerated backends (Transformer Engine, FlashAttention, Mamba, grouped-GEMM/MoE, DeepEP) for better performance — these source-built kernels come from the `compiled` (Hopper/Blackwell) or `compiled-a100` (A100) extras, built by `docker/Dockerfile` (`GPU_TARGET=h100plus` / `a100`). See the [installation guide](https://docs.nvidia.com/nemo/speech/nightly/) for the full list and build details.

### Docker (turnkey, our supported stack)

The latest prebuilt NeMo Speech image is the
[26.07.00 NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-speech?version=26.07.00):

```bash
docker pull nvcr.io/nvidia/nemo-speech:26.07.00
docker run --rm -it --gpus all -v "$PWD:/workspace" nvcr.io/nvidia/nemo-speech:26.07.00 bash
```

To build the container from source (CUDA 13 / H100+ by default):

```bash
git clone https://github.com/NVIDIA-NeMo/Speech.git
cd Speech
docker buildx build -f docker/Dockerfile -t nemo-speech .          # CUDA 13 / H100+ (default)
docker run --rm -it --gpus all -v "$PWD:/workspace" nemo-speech bash
```

For A100, set `GPU_TARGET=a100` — A100 works with **both CUDA 12 and CUDA 13** (CUDA 13, the default base image, is recommended; the CUDA 12 base is a convenience). See the header of [`docker/Dockerfile`](docker/Dockerfile) for all build arguments (`BASE_IMAGE`, `GPU_TARGET`).

### From PyPI with pip (fallback — bring your own versions)

Prefer your own Python/PyTorch/CUDA? Install your PyTorch first (any version ≥ 2.7 for your CPU/CUDA/etc. target — see the [PyTorch install matrix](https://pytorch.org/get-started/locally/)), then add NeMo Speech and it **keeps your build**. `uv pip` (uv's fast, pip-compatible installer) works like `pip`:

```bash
uv pip install 'nemo-toolkit[asr,tts]'   # or plain: pip install 'nemo-toolkit[asr,tts]'
```

> ⚠️ Do **not** use `uv sync --locked` for a bring-your-own stack — it applies `uv.lock` and replaces your Python/PyTorch/CUDA with the supported baseline. Use `uv pip`/`pip` here; reserve `uv sync --locked` for reproducing our stack.

To instead pull *our* pinned PyTorch build, add the CUDA extra and the matching wheel index (pip/uv pip do not read uv's project index config, so `--extra-index-url` is required):

```bash
pip install 'nemo-toolkit[asr,tts,cu13]' --extra-index-url https://download.pytorch.org/whl/cu132   # CUDA 13.x
pip install 'nemo-toolkit[asr,tts,cu12]' --extra-index-url https://download.pytorch.org/whl/cu129   # CUDA 12.x
```

## Contribute to NeMo Speech

We welcome community contributions! Please refer to
[CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/Speech/blob/main/CONTRIBUTING.md) for the process.

## Licenses

NeMo Speech is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Speech/blob/main/LICENSE).
