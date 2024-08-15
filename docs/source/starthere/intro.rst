Introduction
============

.. # define a hard line break for html
.. |br| raw:: html

    <br />

.. _dummy_header:

NVIDIA NeMo Framework is an end-to-end, cloud-native framework for building, customizing, and deploying generative AI models anywhere. It allows for the creation of state-of-the-art models across a wide array of domains, including speech, language, and vision. For detailed information on utilizing NeMo in your generative AI workflows, refer to the `NeMo Framework User Guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html>`_.

Training generative AI architectures typically requires significant data and computing resources. NeMo utilizes `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ for efficient and performant multi-GPU/multi-node mixed-precision training.
NeMo is built on top of NVIDIA's powerful Megatron-LM and Transformer Engine for its Large Language Models (LLMs) and Multimodal Models (MMs), leveraging cutting-edge advancements in model training and optimization. For Speech AI applications, Automatic Speech Recognition (ASR) and Text-to-Speech (TTS), NeMo is developed with native PyTorch and PyTorch Lightning, ensuring seamless integration and ease of use. Future updates are planned to align Speech AI models with the Megatron framework, enhancing training efficiency and model performance.


`NVIDIA NeMo Framework <https://github.com/NVIDIA/NeMo>`_ features separate collections for Large Language Models (LLMs), Multimodal Models (MMs), Computer Vision (CV), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS) models. Each collection comprises prebuilt modules that include everything needed to train on your data. These modules can be easily customized, extended, and composed to create new generative AI model architectures.

Pre-trained NeMo models are available to download on `NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`__ and `HuggingFace Hub <https://huggingface.co/nvidia>`__.

Prerequisites
-------------

Before using NeMo, make sure you meet the following prerequisites:

#. Python version 3.10 or above.

#. Pytorch version 1.13.1 or 2.0+.

#. Access to an NVIDIA GPU for model training.

Installation
------------

**Using NVIDIA PyTorch Container**

To leverage all optimizations for LLM training, including 3D Model Parallel, fused kernels, FP8, and more, we recommend using the NVIDIA PyTorch container.

.. code-block:: bash

    docker pull nvcr.io/nvidia/pytorch:24.01-py3
    docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3

Within the container, you can install NeMo and its dependencies as follows:

NeMo Installation

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']

Transformer Engine Installation

This step involves cloning the Transformer Engine repository, checking out a specific commit, and installing it with specific flags.

.. code-block:: bash

    git clone https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    git fetch origin 8c9abbb80dba196f086b8b602a7cf1bce0040a6a && \
    git checkout FETCH_HEAD && \
    git submodule init && git submodule update && \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

Apex Installation

This step includes a bug fix for Apex in the PyTorch 23.11 container.

.. code-block:: bash

    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    git checkout c07a4cf67102b9cd3f97d1ba36690f985bae4227 && \
    cp -R apex /usr/local/lib/python3.10/dist-packages

PyTorch Lightning Installation

This step involves installing a bug-fixed version of PyTorch Lightning from a specific branch.

.. code-block:: bash

    git clone -b bug_fix https://github.com/athitten/pytorch-lightning.git && \
    cd pytorch-lightning && \
    PACKAGE_NAME=pytorch pip install -e .

Megatron Core Installation

This section details the steps to clone and install the Megatron Core.

.. code-block:: bash

    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    git checkout a5415fcfacef2a37416259bd38b7c4b673583675 && \
    pip install .

TensorRT Model Optimizer Installation

This final step involves installing the TensorRT Model Optimizer package.

.. code-block:: bash

    pip install nvidia-modelopt[torch]~=0.15.0 --extra-index-url https://pypi.nvidia.com


.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']

**Conda Installation**

If you do not use the NVIDIA PyTorch container, we recommend installing NeMo in a clean Conda environment.

.. code-block:: bash

    conda create --name nemo python==3.10.12
    conda activate nemo

Refer to the PyTorch configurator for instructions on installing PyTorch. `configurator <https://pytorch.org/get-started/locally/>`_

Quick Start Guide
-----------------

To explore NeMo's capabilities in LLM, ASR, and TTS, follow the example below based on the `Audio Translation <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/AudioTranslationSample.ipynb>`_ tutorial. Ensure NeMo is :ref:`installed <installation>` before proceeding.


.. code-block:: python

    # Import NeMo's ASR, NLP and TTS collections
    import nemo.collections.asr as nemo_asr
    import nemo.collections.nlp as nemo_nlp
    import nemo.collections.tts as nemo_tts

    # Download an audio file that we will transcribe, translate, and convert the written translation to speech
    import wget
    wget.download("https://nemo-public.s3.us-east-2.amazonaws.com/zh-samples/common_voice_zh-CN_21347786.mp3")

    # Instantiate a Mandarin speech recognition model and transcribe an audio file.
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_zh_citrinet_1024_gamma_0_25")
    mandarin_text = asr_model.transcribe(['common_voice_zh-CN_21347786.mp3'])
    print(mandarin_text)

    # Instantiate Neural Machine Translation model and translate the text
    nmt_model = nemo_nlp.models.MTEncDecModel.from_pretrained(model_name="nmt_zh_en_transformer24x6")
    english_text = nmt_model.translate(mandarin_text)
    print(english_text)

    # Instantiate a spectrogram generator (which converts text -> spectrogram)
    # and vocoder model (which converts spectrogram -> audio waveform)
    spectrogram_generator = nemo_tts.models.FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained(model_name="tts_en_hifigan")

    # Parse the text input, generate the spectrogram, and convert it to audio
    parsed_text = spectrogram_generator.parse(english_text[0])
    spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed_text)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    # Save the audio to a file
    import soundfile as sf
    sf.write("output_audio.wav", audio.to('cpu').detach().numpy()[0], 22050)

For detailed tutorials and documentation on specific tasks or to learn more about NeMo, check out the NeMo :doc:`tutorials <./tutorials>` or dive deeper into the documentation, such as learning about ASR in :doc:`here <../asr/intro>`.

Discussion Board
----------------

For additional information and questions, visit the `NVIDIA NeMo Discussion Board <https://github.com/NVIDIA/NeMo/discussions>`_.

Contribute to NeMo
------------------

Community contributions are welcome! See the `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_ file for how to contribute.

License
-------

NeMo is released under the `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.
