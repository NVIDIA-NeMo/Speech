
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
        <summary>
          <a href="https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/">
            NVIDIA sets new generative AI performance and scale records in MLPerf Training v4.0
          </a> (2024/06/12)
        </summary>

        Using NVIDIA NeMo Framework and NVIDIA Hopper GPUs NVIDIA was able to scale to 11,616 H100 GPUs and achieve near-linear performance scaling on LLM pretraining. 
        NVIDIA also achieved the highest LLM fine-tuning performance and raised the bar for text-to-image training.
        <br><br>
      </details>

      <details>
          <summary>
            <a href="https://cloud.google.com/blog/products/compute/gke-and-nvidia-nemo-framework-to-train-generative-ai-models">
              Accelerate your generative AI journey with NVIDIA NeMo Framework on GKE
            </a> (2024/03/16)
          </summary>

          An end-to-end walkthrough to train generative AI models on the Google Kubernetes Engine (GKE) using the NVIDIA NeMo Framework is available at https://github.com/GoogleCloudPlatform/nvidia-nemo-on-gke. 
          The walkthrough includes detailed instructions on how to set up a Google Cloud Project and pre-train a GPT model using the NeMo Framework.
          <br><br>
        </details>

      <details>
        <summary>
          <a href="https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/">
            Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso
          </a> (2024/03/06)
        </summary>

        Bria, a Tel Aviv startup at the forefront of visual generative AI for enterprises now leverages the NVIDIA NeMo Framework. 
        The Bria.ai platform uses reference implementations from the NeMo Multimodal collection, trained on NVIDIA Tensor Core GPUs, to enable high-throughput and low-latency image generation. 
        Bria has also adopted NVIDIA Picasso, a foundry for visual generative AI models, to run inference.
        <br><br>
      </details>

      <details>
        <summary>
          <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/">
            New NVIDIA NeMo Framework Features and NVIDIA H200
          </a> (2023/12/06)
        </summary>

        NVIDIA NeMo Framework now includes several optimizations and enhancements, 
        including: 
        1) Fully Sharded Data Parallelism (FSDP) to improve the efficiency of training large-scale AI models, 
        2) Mix of Experts (MoE)-based LLM architectures with expert parallelism for efficient LLM training at scale, 
        3) Reinforcement Learning from Human Feedback (RLHF) with TensorRT-LLM for inference stage acceleration, and 
        4) up to 4.2x speedups for Llama 2 pre-training on NVIDIA H200 Tensor Core GPUs.
        <br><br>
        <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility">
        <img src="https://github.com/sbhavani/TransformerEngine/blob/main/docs/examples/H200-NeMo-performance.png" alt="H200-NeMo-performance" style="width: 600px;"></a>
        <br><br>
      </details>

      <details>
        <summary>
          <a href="https://blogs.nvidia.com/blog/nemo-amazon-titan/">
            NVIDIA now powers training for Amazon Titan Foundation models
          </a> (2023/11/28)
        </summary>

        NVIDIA NeMo Framework now empowers the Amazon Titan foundation models (FM) with efficient training of large language models (LLMs). 
        The Titan FMs form the basis of Amazon’s generative AI service, Amazon Bedrock. 
        The NeMo Framework provides a versatile framework for building, customizing, and running LLMs.
        <br><br>
      </details>

  </details>

  <details open>
    <summary><b>Speech Recognition</b></summary>
      <details>
        <summary>
          <a href="https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/">
            New Standard for Speech Recognition and Translation from the NVIDIA NeMo Canary Model
          </a> (2024/04/18)
        </summary>

        The NeMo team just released Canary, a multilingual model that transcribes speech in English, Spanish, German, and French with punctuation and capitalization. 
        Canary also provides bi-directional translation, between English and the three other supported languages.
        <br><br>
      </details>

      <details>
        <summary>
          <a href="https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/">
            Pushing the Boundaries of Speech Recognition with NVIDIA NeMo Parakeet ASR Models
          </a> (2024/04/18)
        </summary>

        NVIDIA NeMo, an end-to-end platform for the development of multimodal generative AI models at scale anywhere—on any cloud and on-premises—released the Parakeet family of automatic speech recognition (ASR) models. 
        These state-of-the-art ASR models, developed in collaboration with Suno.ai, transcribe spoken English with exceptional accuracy.
        <br><br>
      </details>

    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/">
          Turbocharge ASR Accuracy and Speed with NVIDIA NeMo Parakeet-TDT
        </a> (2024/04/18)
      </summary>

      NVIDIA NeMo, an end-to-end platform for developing multimodal generative AI models at scale anywhere—on any cloud and on-premises—recently released Parakeet-TDT. 
      This new addition to the  NeMo ASR Parakeet model family boasts better accuracy and 64% greater speed over the previously best model, Parakeet-RNNT-1.1B.
      <br><br>
    </details>

  </details>

   


Introduction
------------

NVIDIA NeMo Framework is a scalable and cloud-native generative AI framework built for researchers and PyTorch developers working on Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text to Speech (TTS), and Computer Vision (CV) domains. It is designed to help you efficiently create, customize, and deploy new generative AI models by leveraging existing code and pre-trained model checkpoints.

For technical documentation, please see the `NeMo Framework User Guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html>`_.

LLMs and MMs Training, Alignment, and Customization
---------------------------------------------------

All NeMo models are trained with `Lightning <https://github.com/Lightning-AI/lightning>`_.
Training is automatically scalable to 1000s of GPUs.

When applicable, NeMo models leverage cutting-edge distributed training techniques, incorporating `parallelism strategies <https://docs.nvidia.com/nemo-framework/user-guide/latest/modeloverview.html>`_ to enable efficient training of very large models. These techniques include Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and Mixed Precision Training with BFloat16 and FP8, as well as others.

NeMo Transformer-based LLMs and MMs utilize `NVIDIA Transformer Engine <https://github.com/NVIDIA/TransformerEngine>`_ for FP8 training on NVIDIA Hopper GPUs, while leveraging `NVIDIA Megatron Core <https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core>`_ for scaling Transformer model training.

NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF). See `NVIDIA NeMo Aligner <https://github.com/NVIDIA/NeMo-Aligner>`_ for more information.

In addition to supervised fine-tuning (SFT), NeMo also supports the latest parameter efficient fine-tuning (PEFT) techniques such as LoRA, P-Tuning, Adapters, and IA3. Refer to the `NeMo Framework User Guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html>`_ for the full list of supported models and techniques.

LLMs and MMs Deployment and Optimization
----------------------------------------

NeMo LLMs and MMs can be deployed and optimized with `NVIDIA NeMo Microservices <https://developer.nvidia.com/nemo-microservices-early-access>`_.

Speech AI
---------

NeMo ASR and TTS models can be optimized for inference and deployed for production use cases with `NVIDIA Riva <https://developer.nvidia.com/riva>`_.

NeMo Framework Launcher
-----------------------

`NeMo Framework Launcher <https://github.com/NVIDIA/NeMo-Megatron-Launcher>`_ is a cloud-native tool that streamlines the NeMo Framework experience. It is used for launching end-to-end NeMo Framework training jobs on CSPs and Slurm clusters. 

The NeMo Framework Launcher includes extensive recipes, scripts, utilities, and documentation for training NeMo LLMs. It also includes the NeMo Framework `Autoconfigurator <https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration>`_, which is designed to find the optimal model parallel configuration for training on a specific cluster.

To get started quickly with the NeMo Framework Launcher, please see the `NeMo Framework Playbooks <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html>`_. The NeMo Framework Launcher does not currently support ASR and TTS training, but it will soon.

Get Started with NeMo Framework
-------------------------------

Getting started with NeMo Framework is easy. State-of-the-art pretrained NeMo models are freely available on `Hugging Face Hub <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_ and
`NVIDIA NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_.
These models can be used to generate text or images, transcribe audio, and synthesize speech in just a few lines of code.

We have extensive `tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html>`_ that
can be run on `Google Colab <https://colab.research.google.com>`_ or with our `NGC NeMo Framework Container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`_. We also have `playbooks <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html>`_ for users who want to train NeMo models with the NeMo Framework Launcher.

For advanced users who want to train NeMo models from scratch or fine-tune existing NeMo models, we have a full suite of `example scripts <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ that support multi-GPU/multi-node training.

Key Features
------------

* `Large Language Models <nemo/collections/nlp/README.md>`_
* `Multimodal <nemo/collections/multimodal/README.md>`_
* `Automatic Speech Recognition <nemo/collections/asr/README.md>`_
* `Text to Speech <nemo/collections/tts/README.md>`_
* `Computer Vision <nemo/collections/vision/README.md>`_

Requirements
------------

* Python 3.10 or above
* Pytorch 1.13.1 or above
* NVIDIA GPU (if you intend to do model training)

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

Install NeMo Framework
----------------------

The NeMo Framework can be installed in a variety of ways, depending on your needs. Depending on the domain, you may find one of the following installation methods more suitable.

* Conda / Pip - Refer to `Conda <#conda>`_ and `Pip <#pip>`_ for installation instructions.

  * This is the recommended method for ASR and TTS domains.
  * When using a Nvidia PyTorch container as the base, this is the recommended method for all domains.

* Docker Containers - Refer to `Docker containers <#docker-containers>`_ for installation instructions.

  * NeMo Framework container - `nvcr.io/nvidia/nemo:24.05`

* LLMs and MMs Dependencies - Refer to `LLMs and MMs Dependencies <#install-llms-and-mms-dependencies>`_ for installation instructions.

**Important: We strongly recommended that you start with a base NVIDIA PyTorch container: nvcr.io/nvidia/pytorch:24.02-py3.**

Conda
^^^^^

Install NeMo in a fresh Conda environment:

.. code-block:: bash

    conda create --name nemo python==3.10.12
    conda activate nemo

Install PyTorch using their `configurator <https://pytorch.org/get-started/locally/>`_:

.. code-block:: bash

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

The command to install PyTorch may depend on your system. Use the configurator linked above to find the right command for your system.

Then, install NeMo via Pip or from Source. We do not provide NeMo on the conda-forge or any other Conda channel.

Pip
^^^

To install the nemo_toolkit, use the following installation method:

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython packaging
    pip install nemo_toolkit['all']

Depending on the shell used, you may need to use the ``"nemo_toolkit[all]"`` specifier instead in the above command.

Pip from a Specific Domain
^^^^^^^^^^^^^^^^^^^^^^^^^^

To install a specific domain of NeMo, you must first install the nemo_toolkit using the instructions listed above. Then, you run the following domain-specific commands:

.. code-block:: bash

    pip install nemo_toolkit['asr']
    pip install nemo_toolkit['nlp']
    pip install nemo_toolkit['tts']
    pip install nemo_toolkit['vision']
    pip install nemo_toolkit['multimodal']

Pip from a Source Branch
^^^^^^^^^^^^^^^^^^^^^^^^

If you want to work with a specific version of NeMo from a particular GitHub branch (e.g main), use the following installation method:

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython packaging
    python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]


Build from Source
^^^^^^^^^^^^^^^^^

If you want to clone the NeMo GitHub repository and contribute to NeMo open-source development work, use the following installation method:

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

If you only want the toolkit without the additional Conda-based dependencies, you can replace ``reinstall.sh`` with ``pip install -e .`` when your PWD is the root of the NeMo repository.

Mac Computers with Apple Silicon
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install NeMo on Mac computers with the Apple M-Series GPU, you need to create a new Conda environment, install PyTorch 2.0 or higher, and then install the nemo_toolkit.

**Important: This method is only applicable to the ASR domain.**

Run the following code:

.. code-block:: shell

    # [optional] install mecab using Homebrew, to use sacrebleu for NLP collection
    # you can install Homebrew here: https://brew.sh
    brew install mecab

    # [optional] install pynini using Conda, to use text normalization
    conda install -c conda-forge pynini

    # install Cython manually
    pip install cython packaging

    # clone the repo and install in development mode
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    pip install 'nemo_toolkit[all]'

    # Note that only the ASR toolkit is guaranteed to work on MacBook - so for MacBook use pip install 'nemo_toolkit[asr]'

Windows Computers
^^^^^^^^^^^^^^^^^

To install the Windows Subsystem for Linux (WSL), run the following code in PowerShell: 

.. code-block:: shell

    wsl --install
    # [note] If you run wsl --install and see the WSL help text, it means WSL is already installed.

To learn more about installing WSL, refer to `Microsoft's official documentation <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

After installing your Linux distribution with WSL, two options are available:

**Option 1:** Open the distribution (Ubuntu by default) from the Start menu and follow the instructions.

**Option 2:** Launch the Terminal application. Download it from `Microsoft's Windows Terminal page <https://learn.microsoft.com/en-us/windows/terminal>`_ if not installed.

Next, follow the instructions for Linux systems, as provided above. For example:

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

RNNT
^^^^

For optimal performance of a Recurrent Neural Network Transducer (RNNT), install the Numba package from Conda.

Run the following code:

.. code-block:: bash

  conda remove numba
  pip uninstall numba
  conda install -c conda-forge numba

Install LLMs and MMs Dependencies
---------------------------------

If you work with the LLM and MM domains, three additional dependencies are required: NVIDIA Apex, NVIDIA Transformer Engine, and NVIDIA Megatron Core. When working with the `main` branch, these dependencies may require a recent commit.

The most recent working versions of these dependencies are here:

.. code-block:: bash

  export apex_commit=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
  export te_commit=bfe21c3d68b0a9951e5716fb520045db53419c5e
  export mcore_commit=fbb375d4b5e88ce52f5f7125053068caff47f93f
  export nv_pytorch_tag=24.02-py3

When using a released version of NeMo, please refer to the `Software Component Versions <https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html>`_ for the correct versions.

PyTorch Container
^^^^^^^^^^^^^^^^^

We recommended that you start with a base NVIDIA PyTorch container: nvcr.io/nvidia/pytorch:24.02-py3.

If starting with a base NVIDIA PyTorch container, you must first launch the container:

.. code-block:: bash

  docker run \
    --gpus all \
    -it \
    --rm \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    nvcr.io/nvidia/pytorch:$nv_pytorch_tag

Next, you need to install the dependencies.

Apex
^^^^

NVIDIA Apex is required for LLM and MM domains. Although Apex is pre-installed in the NVIDIA PyTorch container, you may need to update it to a newer version.

To install Apex, run the following code:

.. code-block:: bash

    git clone https://github.com/NVIDIA/apex.git
    cd apex
    git checkout $apex_commit
    pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"

When attempting to install Apex separately from the NVIDIA PyTorch container, you might encounter an error if the CUDA version on your system is different from the one used to compile PyTorch. To bypass this error, you can comment out the relevant line in the setup file located in the Apex repository on GitHub here: https://github.com/NVIDIA/apex/blob/master/setup.py#L32.

cuda-nvprof is needed to install Apex. The version should match the CUDA version that you are using.

To install cuda-nvprof, run the following code:

.. code-block:: bash

  conda install -c nvidia cuda-nvprof=11.8

Finally, install the packaging:

.. code-block:: bash

  pip install packaging

To install the most recent versions of Apex locally, it might be necessary to remove the `pyproject.toml` file from the Apex directory.

Transformer Engine
^^^^^^^^^^^^^^^^^^

NVIDIA Transformer Engine is required for LLM and MM domains. Although the Transformer Engine is pre-installed in the NVIDIA PyTorch container, you may need to update it to a newer version.

The Transformer Engine facilitates training with FP8 precision on NVIDIA Hopper GPUs and introduces many enhancements for the training of Transformer-based models. Refer to `Transformer Enginer <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_ for information. 

To install Transformer Engine, run the following code:

.. code-block:: bash

  git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git checkout $te_commit && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

Transformer Engine requires PyTorch to be built with at least CUDA 11.8.

Megatron Core
^^^^^^^^^^^^^

Megatron Core is required for LLM and MM domains. Megatron Core is a library for scaling large Transformer-based models. NeMo LLMs and MMs leverage Megatron Core for model parallelism, transformer architectures, and optimized PyTorch datasets.

To install Megatron Core, run the following code:

.. code-block:: bash

  git clone https://github.com/NVIDIA/Megatron-LM.git && \
  cd Megatron-LM && \
  git checkout $mcore_commit && \
  pip install . && \
  cd megatron/core/datasets && \
  make

NeMo Text Processing
--------------------

NeMo Text Processing, specifically Inverse Text Normalization, is now a separate repository. It is located here: `https://github.com/NVIDIA/NeMo-text-processing <https://github.com/NVIDIA/NeMo-text-processing>`_.

Docker Containers
-----------------

NeMo containers are launched concurrently with NeMo version updates. NeMo Framework now supports LLMs, MMs, ASR, and TTS in a single consolidated Docker container. You can find additional information about released containers on the `NeMo releases page <https://github.com/NVIDIA/NeMo/releases>`_.

To use a pre-built container, run the following code:

.. code-block:: bash

    docker pull nvcr.io/nvidia/nemo:24.05

To build a nemo container with Dockerfile from a branch, run the following code:

.. code-block:: bash

    DOCKER_BUILDKIT=1 docker build -f Dockerfile -t nemo:latest

If you choose to work with the main branch, we recommend using NVIDIA's PyTorch container version 23.10-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:23.10-py3


Future Work
-----------

The NeMo Framework Launcher does not currently support ASR and TTS training, but it will soon.

Discussions Board
-----------------

FAQ can be found on the NeMo `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions on the board.

Contribute to NeMo
------------------

We welcome community contributions! Please refer to `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_ for the process.

Publications
------------------

We provide an ever-growing list of `publications <https://nvidia.github.io/NeMo/publications/>`_ that utilize the NeMo Framework.

To contribute an article to the collection, please submit a pull request to the ``gh-pages-src`` branch of this repository. For detailed information, please consult the README located at the `gh-pages-src branch <https://github.com/NVIDIA/NeMo/tree/gh-pages-src#readme>`_.

Licenses
--------

* `NeMo GitHub Apache 2.0 license <https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file#readme>`__

* NeMo is licensed under the `NVIDIA AI PRODUCT AGREEMENT <https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/>`__. By pulling and using the container, you accept the terms and conditions of this license.
