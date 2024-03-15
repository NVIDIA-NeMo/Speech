
|status| |documentation| |codeql| |license| |pypi| |pyversion| |downloads| |black|

.. |status| image:: http://www.repostatus.org/badges/latest/active.svg
  :target: http://www.repostatus.org/#active
  :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |documentation| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg
  :target: https://github.com/NVIDIA/NeMo/blob/master/LICENSE
  :alt: NeMo core license and license for collections in this repo

.. |pypi| image:: https://badge.fury.io/py/nemo-toolkit.svg
  :target: https://badge.fury.io/py/nemo-toolkit
  :alt: Release version

.. |pyversion| image:: https://img.shields.io/pypi/pyversions/nemo-toolkit.svg
  :target: https://badge.fury.io/py/nemo-toolkit
  :alt: Python version

.. |downloads| image:: https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads
  :target: https://pepy.tech/project/nemo-toolkit
  :alt: PyPi total downloads

.. |codeql| image:: https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push
  :target: https://github.com/nvidia/nemo/actions/workflows/codeql.yml
  :alt: CodeQL

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black
  :alt: Code style: black

.. _main-readme:

**NVIDIA NeMo Framework**
===============

Latest News
-----------

- 2023/12/06 `New NVIDIA NeMo Framework Features and NVIDIA H200 <https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/>`_

.. image:: https://github.com/sbhavani/TransformerEngine/blob/main/docs/examples/H200-NeMo-performance.png
  :target: https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility
  :alt: H200-NeMo-performance
  :width: 600

NeMo Framework has been updated with state-of-the-art features,
such as FSDP, Mixture-of-Experts, and RLHF with TensorRT-LLM to provide speedups up to 4.2x for Llama-2 pre-training on H200.
**All of these features will be available in an upcoming release.**



Introduction
------------

NVIDIA NeMo Framework is a generative AI framework built for researchers and pytorch developers
working on large language models (LLMs), multimodal models (MM), automatic speech recognition (ASR),
and text-to-speech synthesis (TTS).
The primary objective of NeMo is to provide a scalable framework for researchers and developers from industry and academia
to more easily implement and design new generative AI models by being able to leverage existing code and pretrained models.

For technical documentation, please see the `NeMo Framework User Guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html>`_.

All NeMo models are trained with `Lightning <https://github.com/Lightning-AI/lightning>`_ and
training is automatically scalable to 1000s of GPUs.

When applicable, NeMo models take advantage of the latest possible distributed training techniques,
including parallelism strategies such as

* data parallelism
* tensor parallelism
* pipeline model parallelism
* fully sharded data parallelism (FSDP)
* sequence parallelism
* context parallelism
* mixture-of-experts (MoE)

and mixed precision training recipes with bfloat16 and FP8 training.

NeMo's Transformer based LLM and Multimodal models leverage `NVIDIA Transformer Engine <https://github.com/NVIDIA/TransformerEngine>`_ for FP8 training on NVIDIA Hopper GPUs
and leverages `NVIDIA Megatron Core <https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core>`_ for scaling transformer model training.

NeMo LLMs can be aligned with state of the art methods such as SteerLM, DPO and Reinforcement Learning from Human Feedback (RLHF),
see `NVIDIA NeMo Aligner <https://github.com/NVIDIA/NeMo-Aligner>`_ for more details.

NeMo LLM and Multimodal models can be deployed and optimized with `NVIDIA Inference Microservices (Early Access) <https://developer.nvidia.com/nemo-microservices-early-access>`_.

NeMo ASR and TTS models can be optimized for inference and deployed for production use-cases with `NVIDIA Riva <https://developer.nvidia.com/riva>`_.

For scaling NeMo LLM and Multimodal training on Slurm clusters or public clouds, please see the `NVIDIA Framework Launcher <https://github.com/NVIDIA/NeMo-Megatron-Launcher>`_.
The NeMo Framework launcher has extensive recipes, scripts, utilities, and documentation for training NeMo LLMs and Multimodal models and also has an `Autoconfigurator <https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration>`_
which can be used to find the optimal model parallel configuration for training on a specific cluster.
To get started quickly with the NeMo Framework Launcher, please see the `NeMo Framework Playbooks <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html>`_
The NeMo Framework Launcher does not currently support ASR and TTS training but will soon.

Getting started with NeMo is simple.
State of the Art pretrained NeMo models are freely available on `HuggingFace Hub <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_ and
`NVIDIA NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_.
These models can be used to generate text or images, transcribe audio, and synthesize speech in just a few lines of code.

We have extensive `tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html>`_ that
can be run on `Google Colab <https://colab.research.google.com>`_ or with our `NGC NeMo Framework Container. <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`_
and we have `playbooks <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html>`_ for users that want to train NeMo models with the NeMo Framework Launcher.

For advanced users that want to train NeMo models from scratch or finetune existing NeMo models
we have a full suite of `example scripts <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ that support multi-GPU/multi-node training.

Key Features
------------

* `Large Language Models <nemo/collections/nlp/README.md>`_
* `Multimodal <nemo/collections/multimodal/README.md>`_
* `Automatic Speech Recognition <nemo/collections/asr/README.md>`_
* `Text to Speech <nemo/collections/tts/README.md>`_
* `Computer Vision <nemo/collections/vision/README.md>`_

Requirements
------------

1) Python 3.10 or above
2) Pytorch 1.13.1 or above
3) NVIDIA GPU, if you intend to do model training

Developer Documentation
-----------------------

.. |main| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |stable| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable
  :alt: Documentation Status
  :scale: 100%
  :target:  https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/

+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Version | Status      | Description                                                                                                                              |
+=========+=============+==========================================================================================================================================+
| Latest  | |main|      | `Documentation of the latest (i.e. main) branch. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/>`_                  |
+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Stable  | |stable|    | `Documentation of the stable (i.e. most recent release) branch. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/>`_ |
+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+


Getting help with NeMo
----------------------
FAQ can be found on NeMo's `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions there.


Installation
------------

The NeMo Framework can be installed in a variety of ways, depending on your needs. Depending on the domain, you may find one of the following installation methods more suitable.

* Conda / Pip - Refer to the `Conda <#conda>`_ and `Pip <#pip>`_ sections for installation instructions.

  * This is recommended for Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) domains.
  * When using a Nvidia PyTorch container as the base, this is the recommended installation method for all domains.

* Docker - Refer to the `Docker containers <#docker-containers>`_ section for installation instructions.

  * This is recommended for Large Language Models (LLM), Multimodal and Vision domains.
  * NeMo LLM & Multimodal Container - `nvcr.io/nvidia/nemo:24.01.01.framework`
  * NeMo Speech Container - `nvcr.io/nvidia/nemo:24.01.speech`

Conda
~~~~~

We recommend installing NeMo in a fresh Conda environment.

.. code-block:: bash

    conda create --name nemo python==3.10.12
    conda activate nemo

Install PyTorch using their `configurator <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

The command used to install PyTorch may depend on your system. Please use the configurator linked above to find the right command for your system.

Pip
~~~
Use this installation mode if you want the latest released version.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']

Depending on the shell used, you may need to use ``"nemo_toolkit[all]"`` instead in the above command.

Pip (Domain Specific)
~~~~~~~~~~~~~~~~~~~~~

To install only a specific domain of NeMo, use the following commands. Note: It is required to install the above pre-requisites before installing a specific domain of NeMo.

.. code-block:: bash

    pip install nemo_toolkit['asr']
    pip install nemo_toolkit['nlp']
    pip install nemo_toolkit['tts']
    pip install nemo_toolkit['vision']
    pip install nemo_toolkit['multimodal']

Pip from source
~~~~~~~~~~~~~~~
Use this installation mode if you want the version from a particular GitHub branch (e.g main).

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]


From source
~~~~~~~~~~~
Use this installation mode if you are contributing to NeMo.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

If you only want the toolkit without additional conda-based dependencies, you may replace ``reinstall.sh``
with ``pip install -e .`` when your PWD is the root of the NeMo repository.

Mac computers with Apple silicon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To install NeMo on Mac with Apple M-Series GPU:

- create a new Conda environment

- install PyTorch 2.0 or higher

- run the following code:

.. code-block:: shell

    # [optional] install mecab using Homebrew, to use sacrebleu for NLP collection
    # you can install Homebrew here: https://brew.sh
    brew install mecab

    # [optional] install pynini using Conda, to use text normalization
    conda install -c conda-forge pynini

    # install Cython manually
    pip install cython

    # clone the repo and install in development mode
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    pip install 'nemo_toolkit[all]'

    # Note that only the ASR toolkit is guaranteed to work on MacBook - so for MacBook use pip install 'nemo_toolkit[asr]'

Windows Computers
~~~~~~~~~~~~~~~~~

One of the options is using Windows Subsystem for Linux (WSL).

To install WSL:

- In PowerShell, run the following code:

.. code-block:: shell

    wsl --install
    # [note] If you run wsl --install and see the WSL help text, it means WSL is already installed.

Learn more about installing WSL at `Microsoft's official documentation <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

After Installing your Linux distribution with WSL:
  - **Option 1:** Open the distribution (Ubuntu by default) from the Start menu and follow the instructions.
  - **Option 2:** Launch the Terminal application. Download it from `Microsoft's Windows Terminal page <https://learn.microsoft.com/en-us/windows/terminal>`_ if not installed.

Next, follow the instructions for Linux systems, as provided above. For example:

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

RNNT
~~~~
Note that RNNT requires numba to be installed from conda.

.. code-block:: bash

  conda remove numba
  pip uninstall numba
  conda install -c conda-forge numba

Apex
~~~~
NeMo LLM Domain training requires NVIDIA Apex to be installed.
Install it manually if not using the NVIDIA PyTorch container.

To install Apex, run

.. code-block:: bash

    git clone https://github.com/NVIDIA/apex.git
    cd apex
    git checkout b496d85fb88a801d8e680872a12822de310951fd
    pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

It is highly recommended to use the NVIDIA PyTorch or NeMo container if having issues installing Apex or any other dependencies.

While installing Apex, it may raise an error if the CUDA version on your system does not match the CUDA version torch was compiled with.
This raise can be avoided by commenting it here: https://github.com/NVIDIA/apex/blob/master/setup.py#L32

cuda-nvprof is needed to install Apex. The version should match the CUDA version that you are using:

.. code-block:: bash

  conda install -c nvidia cuda-nvprof=11.8

packaging is also needed:

.. code-block:: bash

  pip install packaging

With the latest versions of Apex, the `pyproject.toml` file in Apex may need to be deleted in order to install locally.


Transformer Engine
~~~~~~~~~~~~~~~~~~
NeMo LLM Domain has been integrated with `NVIDIA Transformer Engine <https://github.com/NVIDIA/TransformerEngine>`_
Transformer Engine enables FP8 training on NVIDIA Hopper GPUs.
`Install <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_ it manually if not using the NVIDIA PyTorch container.

.. code-block:: bash

  pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@stable

It is highly recommended to use the NVIDIA PyTorch or NeMo container if having issues installing Transformer Engine or any other dependencies.

Transformer Engine requires PyTorch to be built with CUDA 11.8.


Flash Attention
~~~~~~~~~~~~~~~
When traning Large Language Models in NeMo, users may opt to use Flash Attention for efficient training. Transformer Engine already supports Flash Attention for GPT models. If you want to use Flash Attention for non-causal models, please install `flash-attn <https://github.com/HazyResearch/flash-attention>`_. If you want to use Flash Attention with attention bias (introduced from position encoding, e.g. Alibi), please also install triton pinned version following the `implementation <https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py#L3>`_.

.. code-block:: bash

  pip install flash-attn
  pip install triton==2.0.0.dev20221202

NLP inference UI
~~~~~~~~~~~~~~~~~~~~
To launch the inference web UI server, please install the gradio `gradio <https://gradio.app/>`_.

.. code-block:: bash

  pip install gradio==3.34.0

NeMo Text Processing
~~~~~~~~~~~~~~~~~~~~
NeMo Text Processing, specifically (Inverse) Text Normalization, is now a separate repository `https://github.com/NVIDIA/NeMo-text-processing <https://github.com/NVIDIA/NeMo-text-processing>`_.

Docker containers
~~~~~~~~~~~~~~~~~
We release NeMo containers alongside NeMo releases. For example, NeMo ``r1.23.0`` comes with container ``nemo:24.01.speech``, you may find more details about released containers in `releases page <https://github.com/NVIDIA/NeMo/releases>`_.

To use built container, please run

.. code-block:: bash

    docker pull nvcr.io/nvidia/nemo:24.01.speech

To build a nemo container with Dockerfile from a branch, please run

.. code-block:: bash

    DOCKER_BUILDKIT=1 docker build -f Dockerfile -t nemo:latest .


If you choose to work with the main branch, we recommend using NVIDIA's PyTorch container version 23.10-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:23.10-py3

Examples
--------

Many examples can be found under the `"Examples" <https://github.com/NVIDIA/NeMo/tree/stable/examples>`_ folder.


Contributing
------------

We welcome community contributions! Please refer to `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_ for the process.

Publications
------------

We provide an ever-growing list of `publications <https://nvidia.github.io/NeMo/publications/>`_ that utilize the NeMo framework.

If you would like to add your own article to the list, you are welcome to do so via a pull request to this repository's ``gh-pages-src`` branch.
Please refer to the instructions in the `README of that branch <https://github.com/NVIDIA/NeMo/tree/gh-pages-src#readme>`_.

License
-------
NeMo is released under an `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.
