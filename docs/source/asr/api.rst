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


.. autoclass:: nemo.collections.asr.models.EncDecRNNTBPEModelWithPrompt
    :show-inheritance:
    :members: transcribe, forward, resolve_prompt_language, set_inference_prompt, initialize_prompt_feature, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModelWithPrompt
    :show-inheritance:
    :members: transcribe, set_inference_prompt, initialize_prompt_feature, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecMultiTalkerRNNTBPEModel
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecClassificationModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecSpeakerLabelModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. _asr-api-modules:

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


.. _transformer-encoder-api:

.. autoclass:: nemo.collections.asr.modules.TransformerEncoder
    :show-inheritance:
    :members:


.. _streaming-transformer-encoder-api:

.. autoclass:: nemo.collections.asr.modules.StreamingTransformerEncoder
    :show-inheritance:
    :members:


.. _sequence-packed-teacher-forcing:

Sequence-packed teacher forcing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TransformerDecoderNM.forward_packed`` (``nemo.collections.asr.modules.transformer.transformer``) runs the Transformer decoder in teacher forcing over token-flat inputs:
the target ids of all sequences in one row, the encoder states of all sequences in one row, and the sequence boundaries as explicit metadata instead of masks.
No padded position is computed, so a batch of mixed lengths costs its valid tokens only.
The result equals the rows of the padded ``forward`` at the valid positions, in batch order.

.. code-block:: python

    import torch

    from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
    from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext

    # encoder_states: (B, T, D) with encoder_lengths: (B,), for example from ConformerEncoder.forward;
    # ConformerEncoder.forward_sequence_packed returns the PackedEncoderActivations directly
    packed_encoder = pack_encoder_output(encoder_states, encoder_lengths)
    encoder_context = PackedAttentionContext(packed_encoder)

    # input_ids: (B, U) with input_lengths: (B,); the valid ids of all sequences, in batch order
    valid = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < input_lengths[:, None]
    decoder_context = PackedAttentionContext.from_lengths(input_lengths)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = decoder.forward_packed(input_ids[valid], decoder_context, packed_encoder.data, encoder_context)
    # hidden: (sum(input_lengths), D); the log probabilities follow as in the padded path

The attention of the packed path runs variable-length FlashAttention, which sets the requirements on CUDA:

* float16 or bfloat16 inputs, in training through ``torch.autocast``;
* a head width (``hidden_size / num_attention_heads``) of at most 256 and divisible by 8;
* a GPU of compute capability 8.0 (Ampere) or newer.

Other CUDA inputs, float32 among them, raise ``NotImplementedError`` instead of falling back to a slower path.
On the CPU the sequences are attended one at a time with dense attention in any dtype, which is the reference of the unit tests.

The packed path covers the training forward and backward and leaves out what the padded ``forward`` does beyond that:

* no incremental decoding: ``return_mems`` raises, the decoder memories of the generators are not formed;
* no attention scores: the ``return_xatt_scores`` output of the cross-attention sublayers is empty, the fused kernels do not form the weights;
* no adapters: a decoder block with an enabled adapter raises;
* the causal mask is the standard one, a ``diagonal`` other than 0 raises;
* learned positional encodings bound the sequence length by ``max_sequence_length`` as in the padded path;
* a sequence without encoder states gets a zero cross-attention context, where the padded ``forward`` attends uniformly over its masked frames.


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


Mixins
------

.. autoclass:: nemo.collections.asr.parts.mixins.mixins.ASRBPEMixin
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.mixins.mixins.ASRModuleMixin
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.mixins.rnnt_restart_mixin.RNNTRestartMixin
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

.. autoclass:: nemo.collections.asr.parts.mixins.multitalker_asr_mixins.SpeakerKernelMixin
    :show-inheritance:
    :members: 

.. _asr-api-datasets:

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


Text-to-Text Datasets
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.data.text_to_text.TextToTextDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.text_to_text.TextToTextIterableDataset
    :show-inheritance:
    :members:


Speaker-Tagged Datasets for Multitalker ASR models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.data.audio_to_text_lhotse_speaker.LhotseSpeechToTextSpkBpeDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.audio_to_diar_label_lhotse.LhotseAudioToSpeechE2ESpkDiarDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.data_simulation.MultiSpeakerSimulator
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.data_simulation.RIRMultiSpeakerSimulator
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

.. _asr-audio-preprocessors:

Audio Preprocessors
-------------------

.. autoclass:: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.AudioToMFCCPreprocessor
    :show-inheritance:
    :members:

.. _asr-api-audio-augmentors:

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

.. autoclass:: nemo.collections.asr.parts.submodules.rnnt_beam_decoding.BeamBatchedRNNTInfer
    :show-inheritance:
    :members:

TDT Decoding
~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.parts.submodules.rnnt_greedy_decoding.GreedyTDTInfer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.rnnt_greedy_decoding.GreedyBatchedTDTInfer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.tdt_beam_decoding.BeamTDTInfer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.submodules.tdt_beam_decoding.BeamBatchedTDTInfer
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

.. autoclass:: nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.RelPositionMultiHeadAttentionAdapter
    :show-inheritance:
    :members:
    :member-order: bysource

.. autoclass:: nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module.PositionalEncodingAdapter
    :show-inheritance:
    :members:
    :member-order: bysource

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

