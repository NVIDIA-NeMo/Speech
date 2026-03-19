Introduction
============

.. # define a hard line break for html
.. |br| raw:: html

    <br />

.. _dummy_header:

NVIDIA NeMo Toolkit is an open-source toolkit for building, customizing, and deploying speech, audio, and multimodal language models. It allows for the creation of state-of-the-art models for speech and audio processing. For detailed information on utilizing NeMo in your generative AI workflows, refer to the `NeMo Framework User Guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html>`_.

Training generative AI architectures typically requires significant data and computing resources. NeMo utilizes `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ for efficient and performant multi-GPU/multi-node mixed-precision training.
For Speech AI applications, Automatic Speech Recognition (ASR) and Text-to-Speech (TTS), NeMo is developed with native PyTorch and PyTorch Lightning, ensuring seamless integration and ease of use.


`NVIDIA NeMo Toolkit <https://github.com/NVIDIA/NeMo>`_ features separate collections for Automatic Speech Recognition (ASR), Text-to-Speech (TTS), Audio Processing, and SpeechLM2 models. Each collection comprises prebuilt modules that include everything needed to train on your data. These modules can be easily customized, extended, and composed to create new generative AI model architectures.

Pre-trained NeMo models are available to download on `NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`__ and `HuggingFace Hub <https://huggingface.co/nvidia>`__.

Prerequisites
-------------

Before using NeMo, make sure you meet the following prerequisites:

#. Python version 3.12 or above.

#. PyTorch version 2.7+.

#. Access to an NVIDIA GPU for model training.

Installation
------------

From PyPI:

.. code-block:: bash

    pip install nemo_toolkit[asr,tts]

Available extras: ``asr``, ``tts``, ``audio``, ``common``.

From source:

.. code-block:: bash

    git clone https://github.com/NVIDIA/NeMo.git
    cd NeMo
    pip install -e '.[asr,tts]'


Quick Start Guide
-----------------

To explore NeMo's capabilities, here are examples for ASR, TTS, speaker diarization, and speech language models.

ASR with Parakeet
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    transcript = asr_model.transcribe(["audio.wav"])
    print(transcript)

Text-to-Speech
~~~~~~~~~~~~~~

.. code-block:: python

    import nemo.collections.tts as nemo_tts

    spectrogram_generator = nemo_tts.models.FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained(model_name="tts_en_hifigan")

    parsed_text = spectrogram_generator.parse("Hello, welcome to NeMo!")
    spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed_text)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    import soundfile as sf
    sf.write("output.wav", audio.to('cpu').detach().numpy()[0], 22050)

Speaker Diarization with Sortformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nemo.collections.asr.models import SortformerEncLabelModel

    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
    diar_model.eval()
    predicted_segments = diar_model.diarize(audio=["meeting.wav"], batch_size=1)
    for segment in predicted_segments[0]:
        print(segment)  # begin_seconds, end_seconds, speaker_index

Speech-to-Text with Canary-Qwen (SpeechLM2/SALM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nemo.collections.speechlm2.models import SALM

    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    answer_ids = model.generate(
        prompts=[[{
            "role": "user",
            "content": f"Transcribe the following: {model.audio_locator_tag}",
            "audio": ["speech.wav"],
        }]],
        max_new_tokens=128,
    )
    print(model.tokenizer.ids_to_text(answer_ids[0].cpu()))

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
