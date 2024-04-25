
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
=========================

Latest News
-----------

.. raw:: html

  <details open>
    <summary><b>Large Language Models and Multimodal</b></summary>
        <details>
          <summary><a href="https://cloud.google.com/blog/products/compute/gke-and-nvidia-nemo-framework-to-train-generative-ai-models">Accelerate your generative AI journey with NVIDIA NeMo Framework on GKE</a> (2024/03/16) </summary>

          An end-to-end walkthrough to train generative AI models on the Google Kubernetes Engine (GKE) using the NVIDIA NeMo Framework is available at https://github.com/GoogleCloudPlatform/nvidia-nemo-on-gke. The walkthrough includes detailed instructions on how to set up a Google Cloud Project and pre-train a GPT model using the NeMo Framework.
          <br><br>
        </details>

      <details>
        <summary><a href="https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/">Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso</a> (2024/03/06) </summary>

        Bria, a Tel Aviv startup at the forefront of visual generative AI for enterprises now leverages the NVIDIA NeMo Framework. The Bria.ai platform uses reference implementations from the NeMo Multimodal collection, trained on NVIDIA Tensor Core GPUs, to enable high-throughput and low-latency image generation. Bria has also adopted NVIDIA Picasso, a foundry for visual generative AI models, to run inference.
        <br><br>
    </details>

    <details>
      <summary><a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/">New NVIDIA NeMo Framework Features and NVIDIA H200</a> (2023/12/06) </summary>

      NVIDIA NeMo Framework now includes several optimizations and enhancements, including: 1) Fully Sharded Data Parallelism (FSDP) to improve the efficiency of training large-scale AI models, 2) Mix of Experts (MoE)-based LLM architectures with expert parallelism for efficient LLM training at scale, 3) Reinforcement Learning from Human Feedback (RLHF) with TensorRT-LLM for inference stage acceleration, and 4) up to 4.2x speedups for Llama 2 pre-training on NVIDIA H200 Tensor Core GPUs.
      <br><br>
      <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility"><img src="https://github.com/sbhavani/TransformerEngine/blob/main/docs/examples/H200-NeMo-performance.png" alt="H200-NeMo-performance" style="width: 600px;"></a>
      <br><br>
    </details>

    <details>
      <summary><a href="https://blogs.nvidia.com/blog/nemo-amazon-titan/">NVIDIA now powers training for Amazon Titan Foundation models</a> (2023/11/28) </summary>

      NVIDIA NeMo Framework now empowers the Amazon Titan foundation models (FM) with efficient training of large language models (LLMs). The Titan FMs form the basis of Amazon’s generative AI service, Amazon Bedrock. The NeMo Framework provides a versatile framework for building, customizing, and running LLMs.
      <br><br>
    </details>

  </details>

   


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

* Docker Containers - Refer to the `Docker containers <#docker-containers>`_ section for installation instructions.

  * This is recommended for Large Language Models (LLM), Multimodal and Vision domains.
  * NeMo LLM & Multimodal Container - `nvcr.io/nvidia/nemo:24.03.framework`
  * NeMo Speech Container - `nvcr.io/nvidia/nemo:24.01.speech`

* LLM and Multimodal Dependencies - Refer to the `LLM and Multimodal dependencies <#llm-and-multimodal-dependencies>`_ section for isntallation instructions.
  * It's higly recommended to start with a base NVIDIA PyTorch container: `nvcr.io/nvidia/pytorch:24.02-py3`

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

LLM and Multimodal Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LLM and Multimodal domains require three additional dependencies: 
NVIDIA Apex, NVIDIA Transformer Engine, and NVIDIA Megatron Core.

When working with the `main` branch these dependencies may require a recent commit.
The most recent working versions of these dependencies are:

.. code-block:: bash

  export apex_commit=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
  export te_commit=bfe21c3d68b0a9951e5716fb520045db53419c5e
  export mcore_commit=fbb375d4b5e88ce52f5f7125053068caff47f93f
  export nv_pytorch_tag=24.02-py3

When using a released version of NeMo, 
please refer to the `Software Component Versions <https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html>`_ 
for the correct versions.

If starting with a base NVIDIA PyTorch container first launch the container:

.. code-block:: bash

  docker run \
    --gpus all \
    -it \
    --rm \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    nvcr.io/nvidia/pytorch:$nv_pytorch_tag

Then install the dependencies:

Apex
~~~~
NeMo LLM Multimodal Domains require that NVIDIA Apex to be installed.
Apex comes installed in the NVIDIA PyTorch container but it's possible that
NeMo LLM and Multimodal may need to be updated to a newer version.

To install Apex, run

.. code-block:: bash

    git clone https://github.com/NVIDIA/apex.git
    cd apex
    git checkout $apex_commit
    pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"


While installing Apex outside of the NVIDIA PyTorch container,
it may raise an error if the CUDA version on your system does not match the CUDA version torch was compiled with.
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

The NeMo LLM Multimodal Domains require that NVIDIA Transformer Engine to be installed.
Transformer Engine comes installed in the NVIDIA PyTorch container but it's possible that
NeMo LLM and Multimodal may need Transformer Engine to be updated to a newer version.

Transformer Engine enables FP8 training on NVIDIA Hopper GPUs and many performance optimizations for transformer-based model training.
Documentation for installing Transformer Engine can be found `here <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_. 

.. code-block:: bash

  git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git checkout $te_commit && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

Transformer Engine requires PyTorch to be built with at least CUDA 11.8.

Megatron Core
~~~~~~~~~~~~~

The NeMo LLM Multimodal Domains require that NVIDIA Megatron Core to be installed.
Megatron core is a library for scaling large transfromer base models. 
NeMo LLM and Multimodal models leverage Megatron Core for model parallelism, 
transformer architectures, and optimized pytorch datasets.

NeMo LLM and Multimodal may need Megatron Core to be updated to a recent version.

.. code-block:: bash

  git clone https://github.com/NVIDIA/Megatron-LM.git && \
  cd Megatron-LM && \
  git checkout $mcore_commit && \
  pip install . && \
  cd megatron/core/datasets && \
  make


NeMo Text Processing
~~~~~~~~~~~~~~~~~~~~
NeMo Text Processing, specifically (Inverse) Text Normalization, is now a separate repository `https://github.com/NVIDIA/NeMo-text-processing <https://github.com/NVIDIA/NeMo-text-processing>`_.

Docker containers
~~~~~~~~~~~~~~~~~
We release NeMo containers alongside NeMo releases. For example, NeMo ``r1.23.0`` comes with container ``nemo:24.01.speech``, you may find more details about released containers in `releases page <https://github.com/NVIDIA/NeMo/releases>`_.

To use a pre-built container, please run

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

We provide an ever-growing list of `publications <https://nvidia.github.io/NeMo/publications/>`_ that utilize the NeMo Framework.

If you would like to add your own article to the list, you are welcome to do so via a pull request to this repository's ``gh-pages-src`` branch.
Please refer to the instructions in the `README of that branch <https://github.com/NVIDIA/NeMo/tree/gh-pages-src#readme>`_.

License
-------
NeMo is released under an `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.
