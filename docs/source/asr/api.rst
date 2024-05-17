NeMo ASR API
============


Model Classes
-------------

.. autoclass:: nemo.collections.asr.models.EncDecCTCModel
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecCTCModelBPE
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecRNNTModel
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecRNNTBPEModel
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecClassificationModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecSpeakerLabelModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.hybrid_asr_tts_models.ASRWithTTSModel
    :show-inheritance:
    :members: from_asr_config, from_pretrained_models, save_asr_model_to, setup_training_data

.. _confidence-ensembles-api:

.. autoclass:: nemo.collections.asr.models.confidence_ensemble.ConfidenceEnsembleModel
    :show-inheritance:
    :members: transcribe

Modules
-------

.. autoclass:: nemo.collections.asr.modules.ConvASREncoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.ConvASRDecoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.ConvASRDecoderClassification
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.SpeakerDecoder
    :show-inheritance:
    :members:

.. _conformer-encoder-api:

.. autoclass:: nemo.collections.asr.modules.ConformerEncoder
    :show-inheritance:
    :members:

.. _squeezeformer-encoder-api:

.. autoclass:: nemo.collections.asr.modules.SqueezeformerEncoder
    :show-inheritance:
    :members:

.. _rnn-encoder-api:

.. autoclass:: nemo.collections.asr.modules.RNNEncoder
    :show-inheritance:
    :members:

.. _rnnt-decoder-api:

.. autoclass:: nemo.collections.asr.modules.RNNTDecoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.StatelessTransducerDecoder
    :show-inheritance:
    :members:

.. _rnnt-joint-api:

.. autoclass:: nemo.collections.asr.modules.RNNTJoint
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.SampledRNNTJoint
    :show-inheritance:
    :members:



Parts
-----

.. autoclass:: nemo.collections.asr.parts.submodules.jasper.JasperBlock
    :show-inheritance:
    :members:


Mixins
------

.. autoclass:: nemo.collections.asr.parts.mixins.mixins.ASRBPEMixin
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.mixins.mixins.ASRModuleMixin
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.mixins.transcription.TranscriptionMixin
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.mixins.transcription.TranscribeConfig
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.mixins.interctc_mixin.InterCTCMixin
    :show-inheritance:
    :members:

Datasets
--------

Character Encoding Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.data.audio_to_text.AudioToCharDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset
    :show-inheritance:
    :members:


Text-to-Text Datasets for Hybrid ASR-TTS models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.data.text_to_text.TextToTextDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.text_to_text.TextToTextIterableDataset
    :show-inheritance:
    :members:


Subword Encoding Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.data.audio_to_text.AudioToBPEDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset
    :show-inheritance:
    :members:

Audio Preprocessors
-------------------

.. autoclass:: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.AudioToMFCCPreprocessor
    :show-inheritance:
    :members:

Audio Augmentors
----------------

.. autoclass:: nemo.collections.asr.modules.SpectrogramAugmentation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.CropOrPadSpectrogramAugmentation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.SpeedPerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.TimeStretchPerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.GainPerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.ImpulsePerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.ShiftPerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.NoisePerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.WhiteNoisePerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.RirAndNoisePerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.preprocessing.perturb.TranscodePerturbation
    :show-inheritance:
    :members:

Miscellaneous Classes
---------------------

CTC Decoding
~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.parts.submodules.ctc_decoding.CTCDecoding
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.ctc_decoding.CTCBPEDecoding
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.ctc_greedy_decoding.GreedyCTCInfer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.ctc_beam_decoding.BeamCTCInfer
    :show-inheritance:
    :members:

RNNT Decoding
~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.parts.submodules.rnnt_decoding.RNNTDecoding
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.rnnt_decoding.RNNTBPEDecoding
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.rnnt_greedy_decoding.GreedyRNNTInfer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.rnnt_greedy_decoding.GreedyBatchedRNNTInfer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.rnnt_beam_decoding.BeamRNNTInfer
    :show-inheritance:
    :members:

Hypotheses
~~~~~~~~~~

.. autoclass:: nemo.collections.asr.parts.utils.rnnt_utils.Hypothesis
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.asr.parts.utils.rnnt_utils.NBestHypotheses
    :show-inheritance:
    :no-members:

Adapter Networks
~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.MultiHeadAttentionAdapter
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.RelPositionMultiHeadAttentionAdapter
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.PositionalEncodingAdapter
    :show-inheritance:
    :members:
    :member-order: bysource

-----

.. autoclass:: nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.RelPositionalEncodingAdapter
    :show-inheritance:
    :members:
    :member-order: bysource


Adapter Strategies
~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.MHAResidualAddAdapterStrategy
    :show-inheritance:
    :members:
    :member-order: bysource
    :undoc-members: adapter_module_names

