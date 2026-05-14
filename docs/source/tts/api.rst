.. _nemo tts collection api:

NeMo TTS API
============

Model Classes
-------------

MagpieTTS (Codec-based TTS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MagpieTTS is an end-to-end TTS model that generates audio codes from transcript and optional context (audio or text). It supports multiple architectures (e.g. multi-encoder context, decoder context) and can be used for standard, long-form, and streaming inference.

.. autoclass:: nemo.collections.tts.models.MagpieTTSModel
    :show-inheritance:
    :members: infer_batch, do_tts, generate_long_form_speech, list_available_models, audio_to_codes, codes_to_audio, has_baked_context_embedding, num_baked_speakers, get_baked_context_embeddings_batch, create_longform_chunk_state, embed_text, prepare_context_tensors

Mel-Spectrogram Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nemo.collections.tts.models.FastPitchModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Vocoders
~~~~~~~~
.. autoclass:: nemo.collections.tts.models.HifiGanModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Codecs
~~~~~~
.. autoclass:: nemo.collections.tts.models.AudioCodecModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Base Classes
----------------

The classes below are the base of the TTS pipeline.

.. autoclass:: nemo.collections.tts.models.base.MelToSpec
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.models.base.SpectrogramGenerator
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.models.base.Vocoder
    :show-inheritance:
    :members:


Dataset Processing Classes
--------------------------
.. autoclass:: nemo.collections.tts.data.dataset.TTSDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.data.dataset.VocoderDataset
    :show-inheritance:
    :members:
