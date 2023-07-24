
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

**NVIDIA NeMo**
===============

Introduction
------------

NVIDIA NeMo is a conversational AI toolkit built for researchers working on automatic speech recognition (ASR), 
text-to-speech synthesis (TTS), large language models (LLMs), and 
natural language processing (NLP).
The primary objective of NeMo is to help researchers from industry and academia to reuse prior work (code and pretrained models) 
and make it easier to create new `conversational AI models <https://developer.nvidia.com/conversational-ai#started>`_.

All NeMo models are trained with `Lightning <https://github.com/Lightning-AI/lightning>`_ and 
training is automatically scalable to 1000s of GPUs. 
Additionally, NeMo Megatron LLM models can be trained up to 1 trillion parameters using tensor and pipeline model parallelism.
NeMo models can be optimized for inference and deployed for production use-cases with `NVIDIA Riva <https://developer.nvidia.com/riva>`_.

Getting started with NeMo is simple.
State of the Art pretrained NeMo models are freely available on `HuggingFace Hub <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_ and
`NVIDIA NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_.
These models can be used to transcribe audio, synthesize speech, or translate text in just a few lines of code.

We have extensive `tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html>`_ that 
can all be run on `Google Colab <https://colab.research.google.com>`_.

For advanced users that want to train NeMo models from scratch or finetune existing NeMo models 
we have a full suite of `example scripts <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ that support multi-GPU/multi-node training.

For scaling NeMo LLM training on Slurm clusters or public clouds, please see the `NVIDIA NeMo Megatron Launcher <https://github.com/NVIDIA/NeMo-Megatron-Launcher>`_.
The NM launcher has extensive recipes, scripts, utilities, and documentation for training NeMo LLMs and also has an `Autoconfigurator <https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration>`_ 
which can be used to find the optimal model parallel configuration for training on a specific cluster.

Also see our `introductory video <https://www.youtube.com/embed/wBgpMf_KQVw>`_ for a high level overview of NeMo.

Key Features
------------

* Speech processing
    * `HuggingFace Space for Audio Transcription (File, Microphone and YouTube) <https://huggingface.co/spaces/smajumdar/nemo_multilingual_language_id>`_
    * `Automatic Speech Recognition (ASR) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html>`_
        * Supported ASR models: `<https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html>`_
            * Jasper, QuartzNet, CitriNet, ContextNet
            * Conformer-CTC, Conformer-Transducer, FastConformer-CTC, FastConformer-Transducer
            * Squeezeformer-CTC and Squeezeformer-Transducer
            * LSTM-Transducer (RNNT) and LSTM-CTC
        * Supports the following decoders/losses:
            * CTC
            * Transducer/RNNT
            * Hybrid Transducer/CTC
            * NeMo Original `Multi-blank Transducers <https://arxiv.org/abs/2211.03541>`_ and `Token-and-Duration Transducers (TDT) <https://arxiv.org/abs/2304.06795>`_
        * Streaming/Buffered ASR (CTC/Transducer) - `Chunked Inference Examples <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_chunked_inference>`_
        * Cache-aware Streaming Conformer with multiple lookaheads - `<https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#cache-aware-streaming-conformer>`_
        * Beam Search decoding
        * `Language Modelling for ASR (CTC and RNNT) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html>`_: N-gram LM in fusion with Beam Search decoding, Neural Rescoring with Transformer
        * `Support of long audios for Conformer with memory efficient local attention <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html#inference-on-long-audio>`_
    * `Speech Classification, Speech Command Recognition and Language Identification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_classification/intro.html>`_: MatchboxNet (Command Recognition), AmberNet (LangID)
    * `Voice activity Detection (VAD) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad>`_: MarbleNet
        * ASR with VAD Inference - `Example <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_vad>`_
    * `Speaker Recognition <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html>`_: TitaNet, ECAPA_TDNN, SpeakerNet
    * `Speaker Diarization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html>`_
        * Clustering Diarizer: TitaNet, ECAPA_TDNN, SpeakerNet
        * Neural Diarizer: MSDD (Multi-scale Diarization Decoder)
    * `Speech Intent Detection and Slot Filling <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_intent_slot/intro.html>`_: Conformer-Transformer
    * `Pretrained models on different languages. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_: English, Spanish, German, Russian, Chinese, French, Italian, Polish, ...
    * `NGC collection of pre-trained speech processing models. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_
* Natural Language Processing
    * `NeMo Megatron pre-training of Large Language Models <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html>`_
    * `Neural Machine Translation (NMT) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/machine_translation/machine_translation.html>`_
    * `Punctuation and Capitalization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html>`_
    * `Token classification (named entity recognition) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/token_classification.html>`_
    * `Text classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_classification.html>`_
    * `Joint Intent and Slot Classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/joint_intent_slot.html>`_
    * `Question answering <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/question_answering.html>`_
    * `GLUE benchmark <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/glue_benchmark.html>`_
    * `Information retrieval <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/information_retrieval.html>`_
    * `Entity Linking <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/entity_linking.html>`_
    * `Dialogue State Tracking <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/sgd_qa.html>`_
    * `Prompt Learning <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/nemo_megatron/prompt_learning.html>`_
    * `NGC collection of pre-trained NLP models. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_nlp>`_
    * `Synthetic Tabular Data Generation <https://developer.nvidia.com/blog/generating-synthetic-data-with-transformers-a-solution-for-enterprise-data-challenges/>`_
* Text-to-Speech Synthesis (TTS):
    * `Documentation <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tts/intro.html#>`_
    * Mel-Spectrogram generators: FastPitch, SSL FastPitch, Mixer-TTS/Mixer-TTS-X, RAD-TTS, Tacotron2
    * Vocoders: HiFiGAN, UnivNet, WaveGlow
    * End-to-End Models: VITS
    * `Pre-trained Model Checkpoints in NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_tts>`_
* `Tools <https://github.com/NVIDIA/NeMo/tree/stable/tools>`_
    * `Text Processing (text normalization and inverse text normalization) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_normalization/intro.html>`_
    * `CTC-Segmentation tool <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/ctc_segmentation.html>`_
    * `Speech Data Explorer <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/speech_data_explorer.html>`_: a dash-based tool for interactive exploration of ASR/TTS datasets
    * `Speech Data Processor <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tools/speech_data_processor.html>`_


Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.

Requirements
------------

1) Python 3.9 or above
2) Pytorch 1.13.1 or above
3) NVIDIA GPU for training

Documentation
-------------

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

Tutorials
---------
A great way to start with NeMo is by checking `one of our tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html>`_.

Getting help with NeMo
----------------------
FAQ can be found on NeMo's `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions there.


Installation
------------

Conda
~~~~~

We recommend installing NeMo in a fresh Conda environment.

.. code-block:: bash

    conda create --name nemo python==3.8.10
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

RNNT
~~~~
Note that RNNT requires numba to be installed from conda.

.. code-block:: bash

  conda remove numba
  pip uninstall numba
  conda install -c conda-forge numba

NeMo Megatron
~~~~~~~~~~~~~
NeMo Megatron training requires NVIDIA Apex to be installed.
Install it manually if not using the NVIDIA PyTorch container.

To install Apex, run

.. code-block:: bash

    git clone https://github.com/NVIDIA/apex.git
    cd apex
    git checkout 57057e2fcf1c084c0fcc818f55c0ff6ea1b24ae2
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./

It is highly recommended to use the NVIDIA PyTorch or NeMo container if having issues installing Apex or any other dependencies.

While installing Apex, it may raise an error if the CUDA version on your system does not match the CUDA version torch was compiled with.
This raise can be avoided by commenting it here: https://github.com/NVIDIA/apex/blob/master/setup.py#L32

cuda-nvprof is needed to install Apex. The version should match the CUDA version that you are using:

.. code-block:: bash

  conda install -c nvidia cuda-nvprof=11.8

packaging is also needed:

.. code-block:: bash

  pip install packaging


Transformer Engine
~~~~~~~~~~~~~~~~~~
NeMo Megatron GPT has been integrated with `NVIDIA Transformer Engine <https://github.com/NVIDIA/TransformerEngine>`_
Transformer Engine enables FP8 training on NVIDIA Hopper GPUs.
`Install <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_ it manually if not using the NVIDIA PyTorch container.

.. code-block:: bash

  pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@stable

It is highly recommended to use the NVIDIA PyTorch or NeMo container if having issues installing Transformer Engine or any other dependencies.

Transformer Engine requires PyTorch to be built with CUDA 11.8.


Flash Attention
~~~~~~~~~~~~~~~~~~~~
Transformer Engine already supports Flash Attention for GPT models. If you want to use Flash Attention for non-causal models or use with attention bias (introduced from position encoding, e.g. Alibi), please install `flash-attn <https://github.com/HazyResearch/flash-attention>`_. 

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

Docker containers:
~~~~~~~~~~~~~~~~~~
We release NeMo containers alongside NeMo releases. For example, NeMo ``r1.19.0`` comes with container ``nemo:23.04``, you may find more details about released containers in `releases page <https://github.com/NVIDIA/NeMo/releases>`_.

To use built container, please run

.. code-block:: bash

    docker pull nvcr.io/nvidia/nemo:23.04

To build a nemo container with Dockerfile from a branch, please run

.. code-block:: bash

    DOCKER_BUILDKIT=1 docker build -f Dockerfile -t nemo:latest .


If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 23.06-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:23.06-py3

Examples
--------

Many examples can be found under the `"Examples" <https://github.com/NVIDIA/NeMo/tree/stable/examples>`_ folder.


Contributing
------------

We welcome community contributions! Please refer to the  `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_ CONTRIBUTING.md for the process.

Publications
------------

We provide an ever growing list of publications that utilize the NeMo framework. Please refer to `PUBLICATIONS.md <https://github.com/NVIDIA/NeMo/tree/stable/PUBLICATIONS.md>`_. We welcome the addition of your own articles to this list !

License
-------
NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.
