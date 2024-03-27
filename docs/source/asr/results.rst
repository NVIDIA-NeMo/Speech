Checkpoints
===========

There are two main ways to load pretrained checkpoints in NeMo:

* Using the :code:`restore_from()` method to load a local checkpoint file (``.nemo``), or
* Using the :code:`from_pretrained()` method to download and set up a checkpoint from NGC.

Refer to the following sections for instructions and examples for each.

Note that these instructions are for loading fully trained checkpoints for evaluation or fine-tuning. For resuming an unfinished
training experiment, use the Experiment Manager to do so by setting the ``resume_if_exists`` flag to ``True``.

Loading Local Checkpoints
-------------------------

NeMo automatically saves checkpoints of a model that is trained in a ``.nemo`` format. Alternatively, to manually save the model at any
point, issue :code:`model.save_to(<checkpoint_path>.nemo)`.

If there is a local ``.nemo`` checkpoint that you'd like to load, use the :code:`restore_from()` method:

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  model = nemo_asr.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

Where the model base class is the ASR model class of the original checkpoint, or the general ``ASRModel`` class.


Hybrid ASR-TTS Models Checkpoints
---------------------------------

:ref:`Hybrid ASR-TTS model <Hybrid-ASR-TTS_model>` is a transparent wrapper for the ASR model, text-to-mel-spectrogram generator, and optional enhancer.
The model is saved as a solid ``.nemo`` checkpoint containing all these parts.
Due to transparency, the ASR model can be extracted after training/finetuning separately by using the ``asr_model`` attribute (NeMo submodel)
:code:`hybrid_model.asr_model.save_to(<asr_checkpoint_path>.nemo)` or by using a wrapper
made for convenience purpose :code:`hybrid_model.save_asr_model_to(<asr_checkpoint_path>.nemo)`


NGC Pretrained Checkpoints
--------------------------

The ASR collection has checkpoints of several models trained on various datasets for a variety of tasks. These checkpoints are
obtainable via NGC `NeMo Automatic Speech Recognition collection <https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr>`_.
The model cards on NGC contain more information about each of the checkpoints available.

The tables below list the ASR models available from NGC. The models can be accessed via the :code:`from_pretrained()` method inside
the ASR Model class. In general, you can load any of these models with code in the following format:

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  model = nemo_asr.models.ASRModel.from_pretrained(model_name="<MODEL_NAME>")

Where the model name is the value under "Model Name" entry in the tables below.

For example, to load the base English QuartzNet model for speech recognition, run:

.. code-block:: python

  model = nemo_asr.models.ASRModel.from_pretrained(model_name="QuartzNet15x5Base-En")

You can also call :code:`from_pretrained()` from the specific model class (such as :code:`EncDecCTCModel`
for QuartzNet) if you need to access a specific model functionality.

If you would like to programmatically list the models available for a particular base class, you can use the
:code:`list_available_models()` method.

.. code-block:: python

  nemo_asr.models.<MODEL_BASE_CLASS>.list_available_models()

Transcribing/Inference
^^^^^^^^^^^^^^^^^^^^^^

To perform inference and transcribe a sample of speech after loading the model, use the ``transcribe()`` method:

.. code-block:: python

    model.transcribe(audio=[list of audio files], batch_size=BATCH_SIZE)

``audio`` can be a string path to a file, a list of string paths to multiple files, a numpy or PyTorch tensor that is an audio file loaded via ``soundfile`` or some other library or even a list of such tensors. This expanded support for inputs to transcription should help users to easily integrate NeMo into their pipelines.

-----

You can do inference on a numpy array that represents an audio signal as follows. Note that it is your responsibility to process the audio to be monochannel and 16KHz sample rate before passing it to the model.

.. code-block:: python

    import torch
    import soundfile as sf

    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(<Model Name>)
    model.eval()

    # Load audio files
    audio_file = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an46-mmap-b.wav")
    audio, sr = sf.read(audio_file, dtype='float32')

    audio_file_2 = os.path.join(test_data_dir, "asr", "train", "an4", "wav", "an104-mrcb-b.wav")
    audio_2, sr = sf.read(audio_file_2, dtype='float32')

    # Mix one numpy array audio segment with torch audio tensor
    audio_2 = torch.from_numpy(audio_2)

    # Numpy array + torch tensor mixed tensor input (for batched inference)
    outputs = model.transcribe([audio, audio_2], batch_size=2)

-----

In order to obtain alignments from CTC or RNNT models (previously called ``logprobs``), you can use the following code:

.. code-block:: python

    hyps = model.transcribe(audio=[list of audio files], batch_size=BATCH_SIZE, return_hypotheses=True)
    logprobs = hyps[0].alignments  # or hyps[0][0].alignments for RNNT

-----

Often times, we want to transcribe a large number of files at once (maybe from a manifest for example). In this case, using ``transcribe()`` directly may be incorrect because it will delay the return of the result until every single sample in the input is processed. One work around is to call transcribe() multiple times, each time using a small subset of the data. This workflow is now supported via a :meth:`~nemo.collections.asr.parts.mixins.transcription.TranscriptionMixin.transcribe_generator`.

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(<Model Name>)

    config = model.get_transcribe_config()
    config.batch_size = 32
    generator = model.transcribe_generator(audio, config)

    for processed_outputs in generator:
        # process a batch of 32 results (or less if last batch does not contain 32 elements)
        ....

For more information, see `nemo.collections.asr.modules <./api.html#modules>`__. For more information on the general ``Transcription API``, please take a look at :class:`~nemo.collections.asr.parts.mixins.transcription.TranscriptionMixin`. The audio files should be 16KHz mono-channel wav files.

-----

Inference with Multi-task Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-task models that use structured prompts require additionl task tokens as input, in which case it is recommended to use manifest as input. Below is an example of using the `nvidia/canary-1b` model:

.. code-block:: python

    from nemo.collections.asr.models import EncDecMultiTaskModel
   
    # load model
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
   
    # update dcode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    # run transcribe
    predicted_text = canary_model.transcribe(
          "<path to input manifest file>",
          batch_size=16,  # batch size to run the inference with
    )

Here the manifest file should be a json file where each line has the following format:

.. code-block:: bash

    {
       "audio_filepath": "/path/to/audio.wav",  # path to the audio file
       "duration": None,  # duration of the audio in seconds, set to `None` to use full audio
       "taskname": "asr",  # use "ast" for speech-to-text translation
       "source_lang": "en",  # language of the audio input, set `source_lang`==`target_lang` for ASR
       "target_lang": "en",  # language of the text output
       "pnc": "yes",  # whether to have PnC output, choices=['yes', 'no']
       "answer": "na", # set to non-dummy strings to calculate WER/BLEU scores 
    }

Note that using manifest allows to specify the task configuration for each audio individually. If we want to use the same task configuration for all the audio files, it can be specified in `transcribe` method directly. 

.. code-block:: python

    canary_model.transcribe(
            audio=[list of audio files],
            batch_size=4,  # batch size to run the inference with
            task="asr",  # use "ast" for speech-to-text translation
            source_lang="en",  # language of the audio input, set `source_lang`==`target_lang` for ASR
            target_lang="en",  # language of the text output
            pnc=True,  # whether to have PnC output, choices=[True, False]
    )

Inference on long audio
^^^^^^^^^^^^^^^^^^^^^^^

In some cases the audio is too long for standard inference, especially if you're using a model such as Conformer, where the time and memory costs of the attention layers scale quadratically with the duration.

There are two main ways of performing inference on long audio files in NeMo:

The first way is to use buffered inference, where the audio is divided into chunks to run on, and the output is merged afterwards.
The relevant scripts for this are contained in `this folder <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/asr_chunked_inference>`_.

The second way, specifically for models with the Conformer/Fast Conformer encoder, is to use local attention, which changes the costs to be linear.
You can train Fast Conformer models with Longformer-style (https://arxiv.org/abs/2004.05150) local+global attention using one of the following configs: CTC config at
``<NeMo_git_root>/examples/asr/conf/fastconformer/fast-conformer-long_ctc_bpe.yaml`` and transducer config at ``<NeMo_git_root>/examples/asr/conf/fastconformer/fast-conformer-long_transducer_bpe.yaml``.
You can also convert any model trained with full context attention to local, though this may result in lower WER in some cases. You can switch to local attention when running the
`transcribe <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/transcribe_speech.py>`_ or `evaluation <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/transcribe_speech.py>`_
scripts in the following way:

.. code-block:: python

    python speech_to_text_eval.py \
        (...other parameters...)  \
        ++model_change.conformer.self_attention_model="rel_pos_local_attn" \
        ++model_change.conformer.att_context_size=[128, 128]

Alternatively, you can change the attention model after loading a checkpoint:

.. code-block:: python

    asr_model = ASRModel.from_pretrained('stt_en_conformer_ctc_large')
    asr_model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[128, 128]
    )

Sometimes, the downsampling module at the earliest stage of the model can take more memory than the actual forward pass since it directly operates on the audio sequence which may not be able to fit in memory for very long audio files. In order to reduce the memory consumption of the subsampling module, you can ask the model to perform auto-chunking of the input sequence and process it piece by piece, taking more time but avoiding an OutOfMemoryError.

.. code-block:: python

    asr_model = ASRModel.from_pretrained('stt_en_fastconformer_ctc_large')
    # Speedup conv subsampling factor to speed up the subsampling module.
    asr_model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select


.. note::

    Only certain models which use depthwise separable convolutions in the downsampling layer support this operation. Please try it out on your model and see if it is supported.


Inference on Apple M-Series GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform inference on Apple Mac M-Series GPU (``mps`` PyTorch device), use PyTorch 2.0 or higher (see the `mac-installation <https://github.com/NVIDIA/NeMo/blob/stable/README.rst#mac-computers-with-apple-silicon>` section). Environment variable ``PYTORCH_ENABLE_MPS_FALLBACK=1`` should be set, since not all operations in PyTorch are currently implemented on ``mps`` device.

If ``allow_mps=true`` flag is passed to ``speech_to_text_eval.py``, the ``mps`` device will be selected automatically.

.. code-block:: python

    PYTORCH_ENABLE_MPS_FALLBACK=1 python speech_to_text_eval.py \
      (...other parameters...)  \
      allow_mps=true


Fine-tuning on Different Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple ASR tutorials provided in the :ref:`Tutorials <tutorials>` section. Most of these tutorials explain how to instantiate a pre-trained model, prepare the model for fine-tuning on some dataset (in the same language) as a demonstration.

Inference Execution Flow Diagram
--------------------------------

When preparing your own inference scripts, please follow the execution flow diagram order for correct inference, found at the `examples directory for ASR collection <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/README.md>`_.

Automatic Speech Recognition Models
-----------------------------------

Below is a list of all the ASR models that are available in NeMo for specific languages, as well as auxiliary language models for certain languages.

Language Models for ASR
^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: data/asrlm_results.csv
   :align: left
   :widths: 30, 30, 40
   :header-rows: 1

|


.. _asr-checkpoint-list-by-language:

Speech Recognition (Languages)
------------------------------

English
^^^^^^^
.. csv-table::
   :file: data/benchmark_en.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Mandarin
^^^^^^^^
.. csv-table::
   :file: data/benchmark_zh.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

German
^^^^^^
.. csv-table::
   :file: data/benchmark_de.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

French
^^^^^^
.. csv-table::
   :file: data/benchmark_fr.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Polish
^^^^^^
.. csv-table::
   :file: data/benchmark_pl.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Italian
^^^^^^^
.. csv-table::
   :file: data/benchmark_it.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Russian
^^^^^^^
.. csv-table::
   :file: data/benchmark_ru.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Spanish
^^^^^^^
.. csv-table::
   :file: data/benchmark_es.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1


-----------------------------

Catalan
^^^^^^^
.. csv-table::
   :file: data/benchmark_ca.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Hindi
^^^^^^^
.. csv-table::
   :file: data/benchmark_hi.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Marathi
^^^^^^^
.. csv-table::
   :file: data/benchmark_mr.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Kinyarwanda
^^^^^^^^^^^
.. csv-table::
   :file: data/benchmark_rw.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Belarusian
^^^^^^^^^^
.. csv-table::
   :file: data/benchmark_by.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Ukrainian
^^^^^^^^^
.. csv-table::
   :file: data/benchmark_ua.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Multilingual
^^^^^^^^^^^^
.. csv-table::
   :file: data/benchmark_multilingual.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------

Code-Switching
^^^^^^^^^^^^^^
.. csv-table::
   :file: data/benchmark_code_switching.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1
