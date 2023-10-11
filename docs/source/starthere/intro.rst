Introduction
============

.. # define a hard line break for html
.. |br| raw:: html

    <br />

.. _dummy_header:

`NVIDIA NeMo <https://github.com/NVIDIA/NeMo>`_, part of the NVIDIA AI platform, is a toolkit for building new state-of-the-art
conversational AI models. NeMo has separate collections for Automatic Speech Recognition (ASR),
Natural Language Processing (NLP), and Text-to-Speech (TTS) models. Each collection consists of
prebuilt modules that include everything needed to train on your data.
Every module can easily be customized, extended, and composed to create new conversational AI
model architectures.

Conversational AI architectures are typically large and require a lot of data and compute
for training. NeMo uses `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ for easy and performant multi-GPU/multi-node
mixed-precision training.

`Pre-trained NeMo models <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_ are available
in 14+ languages.

Prerequisites
-------------

Before you begin using NeMo, it's assumed you meet the following prerequisites.

#. You have Python version 3.10 or above.

#. You have Pytorch version 1.13.1 or 2.0+.

#. You have access to an NVIDIA GPU, if you intend to do model training.

.. _quick_start_guide:

Quick Start Guide
-----------------

You can try out NeMo's ASR, NLP and TTS functionality with the example below, which is based on the `Audio Translation <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/AudioTranslationSample.ipynb>`_ tutorial.

Once you have :ref:`installed NeMo <installation>`, then you can run the code below:

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

You can learn more by about specific tasks you are interested in by checking out the NeMo :doc:`tutorials <./tutorials>`, or documentation (e.g. read :doc:`here <../asr/intro>` to learn more about ASR).

You can also learn more about NeMo in the `NeMo Primer <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/00_NeMo_Primer.ipynb>`_ tutorial, which introduces NeMo, PyTorch Lightning, and OmegaConf, and shows how to use, modify, save, and restore NeMo models. Additionally, the `NeMo Models <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/01_NeMo_Models.ipynb>`__ tutorial explains the fundamentals of how NeMo models are created. These concepts are also explained in detail in the :doc:`NeMo Core <../core/core>` documentation.


Introductory videos
-------------------

See the two introductory videos below for a high level overview of NeMo.

**Developing State-Of-The-Art Conversational AI Models in Three Lines of Code**

.. raw:: html

    <div style="position: relative; padding-bottom: 3%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/wBgpMf_KQVw" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

**NVIDIA NeMo: Toolkit for Conversational AI at PyData Yerevan 2022**

.. raw:: html

    <div style="position: relative; padding-bottom: 3%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/J-P6Sczmas8?mute=0&start=14&autoplay=0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

.. _installation:

Installation
------------

The simplest way to install NeMo is via pip, see info below. 

.. note:: Full NeMo installation instructions (with more ways to install NeMo, and how to handle optional dependencies) can be found in the `GitHub README <https://github.com/NVIDIA/NeMo#installation>`_.

Conda
~~~~~

We recommend installing NeMo in a fresh Conda environment.

.. code-block:: bash

    conda create --name nemo python==3.10.12
    conda activate nemo

Install PyTorch using their `configurator <https://pytorch.org/get-started/locally/>`_.

Pip
~~~
Use this installation mode if you want the latest released version.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']

Depending on the shell used, you may need to use ``"nemo_toolkit[all]"`` instead in the above command.

Discussion board
----------------
For more information and questions, visit the `NVIDIA NeMo Discussion Board <https://github.com/NVIDIA/NeMo/discussions>`_.

Contributing
------------

We welcome community contributions! Refer to the `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_  file for the process.

License
-------

NeMo is released under an `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.