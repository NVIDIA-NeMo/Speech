[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/nemo/speech/nightly/starthere/intro.html)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# **NVIDIA NeMo Speech**
Checkout our [HuggingFace🤗 collection](https://huggingface.co/collections/nvidia/nemotron-speech) for the latest open
weight checkpoints and demos!

## Updates

- 2026-03: [Nemotron-Speech-Streaming v2603](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) has been
    updated. It has been trained on a larger and more diverse corpus, resulting in lower WER across all latency modes.
    Try out [the demo](https://huggingface.co/spaces/nvidia/nemotron-speech-streaming-en-0.6b) and check out
    [the NIM](https://build.nvidia.com/nvidia/nemotron-asr-streaming).
- 2026-03: [MagpieTTS v2602](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) has been released with support
    for 9 languages(En, Es, De, Fr, Vi, It, Zh, Hi, Ja). Try out
    [the demo](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) and check out
    [the NIM](https://build.nvidia.com/nvidia/magpie-tts-multilingual).
- 2026-01: Nemotron-Speech-Streaming was released: One checkpoint that enables users to pick their optimal point
    on the latency-accuracy Pareto curve!
- 2026-01: MagpieTTS was released.
- 2026: This repo has pivoted to focus on audio, speech, and multimodal LLM. For the last NeMo release with support for more
    modalities, see [v2.7.0](https://github.com/NVIDIA-NeMo/NeMo/releases/tag/v2.7.0)
- 2025-08: [Parakeet V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and
    [Canary V2](https://huggingface.co/nvidia/canary-1b-v2) have been released with speech recognition and translation
    support for 25 European languages.
- 2025-06: [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) has been released with record-setting
    5.63% WER on English Open ASR Leaderboard.

## Introduction

NVIDIA NeMo Speech is built for researchers and PyTorch developers working on Speech models including Automatic Speech
Recognition (ASR), Text to Speech (TTS), and Speech LLMs. It is designed to help you efficiently create, customize, and
deploy new It is designed to help you efficiently create, customize, and deploy new AI models by leveraging existing
code and pre-trained model checkpoints.

For technical documentation, please see the
[NeMo Framework User Guide](https://docs.nvidia.com/nemo/speech/nightly/starthere/intro.html).

## Requirements

- Python 3.12 or above
- Pytorch 2.6 or above
- NVIDIA GPU (if you intend to do model training)

As of [Pytorch 2.6](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true),
`torch.load` defaults to using `weights_only=True`. Some model checkpoints may require using `weights_only=False`.
In this case, you can set the env var `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before running code that uses `torch.load`.
However, this should only be done with trusted files. Loading files from untrusted sources with more than weights only
can have the risk of arbitrary code execution.

## Developer Documentation

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/nemo/speech/nightly/starthere/intro.html)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | Documentation of the stable (i.e. most recent release) - To be added |

## Install NeMo Speech

NeMo Speech is installable via pip: `pip install 'nemo-toolkit[all]'`
To install with extra dependencies for CUDA 12.x or 13.x, use `pip install 'nemo-toolkit[all,cu12]'` 
or `pip install 'nemo-toolkit[all,cu13]'` respectively.

## Contribute to NeMo

We welcome community contributions! Please refer to
[CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/NeMo/blob/main/CONTRIBUTING.md) for the process.

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
