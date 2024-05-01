Models
======

This section gives a brief overview of the models that NeMo's ASR collection currently supports.

Each of these models can be used with the example ASR scripts (in the ``<NeMo_git_root>/examples/asr`` directory) by
specifying the model architecture in the config file used. Examples of config files for each model can be found in
the ``<NeMo_git_root>/examples/asr/conf`` directory.

For more information about the config files and how they should be structured, refer to the :doc:`./configs` section.

Pretrained checkpoints for all of these models, as well as instructions on how to load them, can be found in the :doc:`./results`
section. You can use the available checkpoints for immediate inference, or fine-tune them on your own datasets. The checkpoints section
also contains benchmark results for the available ASR models.


Spotlight Models
----------------

Canary
~~~~~~

Canary-1B is the latest ASR model from NVIDIA NeMo. It sits at the top of the `HuggingFace OpenASR Leaderboard <https://huggingface.co/spaces/hf-audio/open_asr_leaderboard>`__ at time of publishing.

You can `download the checkpoint <https://huggingface.co/nvidia/canary-1b>`__  or try out Canary in action in this `HuggingFace Space <https://huggingface.co/spaces/nvidia/canary-1b>`__.

Canary-1B is an encoder-decoder model with a :ref:`FastConformer Encoder <Fast-Conformer>` and Transformer Decoder :cite:`asr-models-vaswani2017aayn`.

It is a multi-lingual, multi-task model, supporting automatic speech-to-text recognition (ASR) in 4 languages (English, German, French, Spanish) as well as translation between English and the 3 other supported languages.


Parakeet
~~~~~~~~

Parakeet is the name of a family of ASR models with a :ref:`FastConformer Encoder <Fast-Conformer>` and a CTC, RNN-T, or TDT decoder.

Model checkpoints:

* `Parakeet-CTC-0.6B <https://huggingface.co/nvidia/parakeet-ctc-0.6b>`__ and `Parakeet-CTC-1.1B <https://huggingface.co/nvidia/parakeet-ctc-1.1b>`__ model cards
* `Parakeet-RNNT-0.6B <https://huggingface.co/nvidia/parakeet-rnnt-0.6b>`__ and `Parakeet-RNNT-1.1B <https://huggingface.co/nvidia/parakeet-rnnt-1.1b>`__ model cards
* `Parakeet-TDT-1.1B <https://huggingface.co/nvidia/parakeet-tdt-1.1b>`__ model card

HuggingFace Spaces to try out Parakeet models in your browser:

* `Parakeet-RNNT-1.1B <https://huggingface.co/spaces/nvidia/parakeet-rnnt-1.1b>`__ space
* `Parakeet-TDT-1.1B <https://huggingface.co/spaces/nvidia/parakeet-tdt-1.1b>`__ space

.. _Conformer_model:

Conformer
---------

.. _Conformer-CTC_model:

Conformer-CTC
~~~~~~~~~~~~~

Conformer-CTC is a CTC-based variant of the Conformer model introduced in :cite:`asr-models-gulati2020conformer`. Conformer-CTC has a
similar encoder as the original Conformer but uses CTC loss and decoding instead of RNNT/Transducer loss, which makes it a non-autoregressive model.
We also drop the LSTM decoder and instead use a linear decoder on the top of the encoder. This model uses the combination of
self-attention and convolution modules to achieve the best of the two approaches, the self-attention layers can learn the global
interaction while the convolutions efficiently capture the local correlations. The self-attention modules support both regular
self-attention with absolute positional encoding, and also Transformer-XL's self-attention with relative positional encodings.

Here is the overall architecture of the encoder of Conformer-CTC:

    .. image:: images/conformer_ctc.png
        :align: center
        :alt: Conformer-CTC Model
        :scale: 50%

This model supports both the sub-word level and character level encodings. You can find more details on the config files for the
Conformer-CTC models in the `Conformer-CTC configuration documentation <./configs.html#conformer-ctc>`_. The variant with sub-word encoding is a BPE-based model
which can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModelBPE` class, while the
character-based variant is based on :class:`~nemo.collections.asr.models.EncDecCTCModel`.

You may find the example config files of Conformer-CTC model with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_bpe.yaml``.

.. _Conformer-Transducer_model:

Conformer-Transducer
~~~~~~~~~~~~~~~~~~~~

Conformer-Transducer is the Conformer model introduced in :cite:`asr-models-gulati2020conformer` and uses RNNT/Transducer loss/decoder.
It has the same encoder as Conformer-CTC but utilizes RNNT/Transducer loss/decoder which makes it an autoregressive model.

Most of the config file for Conformer-Transducer models are similar to Conformer-CTC except the sections related to the decoder and loss: decoder, loss, joint, decoding.
You may take a look at our `tutorials page <../starthere/tutorials.html>`_ on Transducer models to become familiar with their configs:
`Introduction to Transducers <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Intro_to_Transducers.ipynb>`_ and
`ASR with Transducers <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_with_Transducers.ipynb>`_
You can find more details on the config files for the Conformer-Transducer models in the `Conformer-CTC configuration documentation <./configs.html#conformer-ctc>`_.

This model supports both the sub-word level and character level encodings. The variant with sub-word encoding is a BPE-based model
which can be instantiated using the :class:`~nemo.collections.asr.models.EncDecRNNTBPEModel` class, while the
character-based variant is based on :class:`~nemo.collections.asr.models.EncDecRNNTModel`.

You may find the example config files of Conformer-Transducer model with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/conformer/conformer_transducer_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_transducer_bpe.yaml``.

.. _Conformer-HAT_model:

Conformer-HAT
~~~~~~~~~~~~~

Conformer HAT (Hybrid Autoregressive Transducer) model (do not confuse it with Hybrid-Transducer-CTC) is a modification of Conformer-Transducer model based on this previous `work <https://arxiv.org/abs/2003.07705>`_.
The main idea is to separate labels and blank score predictions, which allows to estimate the internal LM probabilities during decoding.
When external LM is available for inference, the internal LM can be subtracted from HAT model prediction in beamsearch decoding to improve external LM efficiency.
It can be helpful in the case of text-only adaptation for new domains.

The only difference from the standard Conformer-Transducer model (RNNT) is the use of `"HATJiont" <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/hybrid_autoregressive_transducer.py#L39>`_
class (instead of "RNNTJoint") for joint module. The all HAT logic is implemented in the "HATJiont" class.

    .. image:: images/hat.png
        :align: center
        :alt: HAT Model
        :scale: 50%

You may find the example config files of Conformer-HAT model with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/conformer/hat/conformer_hat_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/conformer/hat/conformer_hat_bpe.yaml``.

By default, the decoding for HAT model works in the same way as for Conformer-Transducer.
In the case of external ngram LM fusion you can use ``<NeMo_git_root>/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_transducer.py``.
To enable HAT internal LM subtraction set ``hat_subtract_ilm=True`` and find more appropriate couple of ``beam_alpha`` and ``hat_ilm_weight`` values in terms of the best recognition accuracy.

.. _Fast-Conformer:

Fast-Conformer
--------------

The Fast Conformer (CTC and RNNT) models have a faster version of the Conformer encoder and differ from it as follows:

* 8x depthwise convolutional subsampling with 256 channels
* Reduced convolutional kernel size of 9 in the conformer blocks

The Fast Conformer encoder is about 2.4x faster than the regular Conformer encoder without a significant model quality degradation.
128 subsampling channels yield a 2.7x speedup vs baseline but model quality starts to degrade.
With local attention, inference is possible on audios >1 hrs (256 subsampling channels) / >2 hrs (128 channels).

Fast Conformer models were trained using CosineAnnealing (instead of Noam) as the scheduler.

You may find the example CTC config at
``<NeMo_git_root>/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml`` and
the transducer config at ``<NeMo_git_root>/examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml``

Note that both configs are subword-based (BPE).

You can also train these models with longformer-style attention (https://arxiv.org/abs/2004.05150) using the following configs: CTC config at
``<NeMo_git_root>/examples/asr/conf/fastconformer/fast-conformer-long_ctc_bpe.yaml`` and transducer config at ``<NeMo_git_root>/examples/asr/conf/fastconformer/fast-conformer-long_transducer_bpe.yaml``
This allows using the model on longer audio (up to 70 minutes with Fast Conformer). Note that the Fast Conformer checkpoints
can be used with limited context attention even if trained with full context. However, if you also want to use global tokens,
which help aggregate information from outside the limited context, then training is required.

You may find more examples under ``<NeMo_git_root>/examples/asr/conf/fastconformer/``.

Cache-aware Streaming Conformer
-------------------------------

Try real-time ASR with the `Cache-aware Streaming Conformer tutorial notebook <https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb>`_.

Buffered streaming uses overlapping chunks to make an offline ASR model usable for streaming with reasonable accuracy. However, it causes significant amount of duplication in computation due to the overlapping chunks.
Also, there is an accuracy gap between the offline model and the streaming one, as there is inconsistency between how we train the model and how we perform inference for streaming.
The Cache-aware Streaming Conformer models tackle and address these disadvantages. These streaming Conformers are trained with limited right context, making it possible to match how the model is being used in both training and inference.
They also use caching to store intermediate activations to avoid any duplication in compute.
The cache-aware approach is supported for both the Conformer-CTC and Conformer-Transducer and enables the model to be used very efficiently for streaming.

Three categories of layers in Conformer have access to right tokens:
#. depthwise convolutions
#. self-attention
#. convolutions in the downsampling layers.

Streaming Conformer models use causal convolutions or convolutions with lower right context and also self-attention with limited right context to limit the effective right context for the input.
The model trained with such limitations can be used in streaming mode and give the exact same outputs and accuracy as when the whole audio is given to the model in offline mode.
These model can use caching mechanism to store and reuse the activations during streaming inference to avoid any duplications in the computations as much as possible.

We support the following three right context modeling techniques:

* | Fully causal model with zero look-ahead: tokens do not see any future tokens. Convolution layers are all causal and right tokens are masked for self-attention.
  |
  | It gives zero latency but with limited accuracy.
  | To train such a model, you need to set `model.encoder.att_context_size=[left_context,0]` and `model.encoder.conv_context_size=causal` in the config.

* | Regular look-ahead: convolutions are able to see few future frames, and self-attention also sees the same number of future tokens.
  |
  | In this approach the activations for the look-ahead part are not cached, and are recalculated in the next chunks. The right context in each layer should be a small number as multiple layers would increase the effective context size and then increase the look-ahead size and latency.
  | For example for a model of 17 layers with 4x downsampling and 10ms window shift, then even 2 right context in each layer means 17*2*10*4=1360ms look-ahead. Each step after the downsampling corresponds to 4*10=40ms.

* | Chunk-aware look-ahead: input is split into equal chunks. Convolutions are fully causal while self-attention layers are able to see all the tokens in their corresponding chunk.
  |
  | For example, in a model with chunk size of 20 tokens, tokens at the first position of each chunk would see all the next 19 tokens while the last token would see zero future tokens.
  | This approach is more efficient than regular look-ahead in terms of computations as the activations for most of the look-ahead part would be cached and there is close to zero duplications in the calculations.
  | In terms of accuracy, this approach gives similar or even better results in term of accuracy than regular look-ahead as each token in each layer have access to more tokens on average. That is why we recommend to use this approach for streaming. Therefore we recommend to use the chunk-aware for cache-aware models.

.. note:: Latencies are based on the assumption that the forward time of the network is zero and it just estimates the time needed after a frame would be available until it is passed through the model.

Approaches with non-zero look-ahead can give significantly better accuracy by sacrificing latency. The latency can get controlled by the left context size. Increasing the right context would help the accuracy to a limit but would increase the computation time.

In all modes, left context can be controlled by the number of tokens visible in self-attention and the kernel size of the convolutions.
For example, if the left context of self-attention in each layer is set to 20 tokens and there are 10 layers of Conformer, then the effective left context is 20*10=200 tokens.
Left context of self-attention for regular look-ahead can be set as any number, while it should be set as a multiple of the right context in chunk-aware look-ahead.
For convolutions, if we use a left context of 30, then there would be 30*10=300 effective left context.
Left context of convolutions is dependent on their kernel size while it can be any number for self-attention layers. Higher left context for self-attention means larger cache and more computations for the self-attention.
A self-attention left context of around 6 secs would give close results to unlimited left context. For a model with 4x downsampling and shift window of 10ms in the preprocessor, each token corresponds to 4*10=40ms.

If striding approach is used for downsampling, all the convolutions in downsampling would be fully causal and don't see future tokens.

Multiple Look-aheads
~~~~~~~~~~~~~~~~~~~~

We support multiple look-aheads for cahce-aware models. You may specify a list of context sizes for att_context_size.
During the training, different context sizes would be used randomly with the distribution specified by att_context_probs.
For example you may enable multiple look-aheads by setting `model.encoder.att_context_size=[[70,13],[70,6],[70,1],[70,0]]` for the training.
The first item in the list would be the default during test/validation/inference. To switch between different look-aheads, you may use the method `asr_model.encoder.set_default_att_context_size(att_context_size)` or set the att_context_size like the following when using the script `speech_transcribe.py`:

.. code-block:: bash

    python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
    pretrained_name="stt_en_fastconformer_hybrid_large_streaming_multi" \
    audio_dir="<DIRECTORY CONTAINING AUDIO FILES>" \
    att_context_size=[70,0]

..

You may find the example config files for cache-aware streaming FastConformer models at
``<NeMo_git_root>/examples/asr/conf/fastconformer/cache_aware_streaming/conformer_transducer_bpe_streaming.yaml`` for Transducer variant and
at ``<NeMo_git_root>/examples/asr/conf/conformer/cache_aware_streaming/conformer_ctc_bpe.yaml`` for CTC variant. It is recommended to use FastConformer as they are more than 2X faster in both training and inference than regular Conformer.
The hybrid versions of FastConformer can be found here: ``<NeMo_git_root>/examples/asr/conf/conformer/hybrid_cache_aware_streaming/``

Examples for regular Conformer can be found at
``<NeMo_git_root>/examples/asr/conf/conformer/cache_aware_streaming/conformer_transducer_bpe_streaming.yaml`` for Transducer variant and
at ``<NeMo_git_root>/examples/asr/conf/conformer/cache_aware_streaming/conformer_ctc_bpe.yaml`` for CTC variant.

To simulate cache-aware streaming, you may use the script at ``<NeMo_git_root>/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py``. It can simulate streaming in single stream or multi-stream mode (in batches) for an ASR model.
This script can be used for models trained offline with full-context but the accuracy would not be great unless the chunk size is large enough which would result in high latency.
It is recommended to train a model in streaming model with limited context for this script. More info can be found in the script.

Note cache-aware streaming models are being exported without caching support by default.
To include caching support, `model.set_export_config({'cache_support' : 'True'})` should be called before export.
Or, if ``<NeMo_git_root>/scripts/export.py`` is being used:
`python export.py cache_aware_conformer.nemo cache_aware_conformer.onnx --export-config cache_support=True`


.. _Hybrid-Transducer_CTC_model:

Hybrid-Transducer-CTC
---------------------

Hybrid RNNT-CTC models is a group of models with both the RNNT and CTC decoders. Training a unified model would speedup the convergence for the CTC models and would enable
the user to use a single model which works as both a CTC and RNNT model. This category can be used with any of the ASR models.
Hybrid models uses two decoders of CTC and RNNT on the top of the encoder. The default decoding strategy after the training is done is RNNT.
User may use the ``asr_model.change_decoding_strategy(decoder_type='ctc' or 'rnnt')`` to change the default decoding.

The variant with sub-word encoding is a BPE-based model
which can be instantiated using the :class:`~nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModel` class, while the
character-based variant is based on :class:`~nemo.collections.asr.models.EncDecHybridRNNTCTCModel`.

You may use the example scripts under ``<NeMo_git_root>/examples/asr/asr_hybrid_transducer_ctc`` for both the char-based encoding and sub-word encoding.
These examples can be used to train any Hybrid ASR model like Conformer, Citrinet, QuartzNet, etc.

You may find the example config files of Conformer variant of such hybrid models with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/conformer/hybrid_transducer_ctc/conformer_hybrid_transducer_ctc_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/conformer/hybrid_transducer_ctc/conformer_hybrid_transducer_ctc_bpe.yaml``.

Similar example configs for FastConformer variants of Hybrid models can be found here:
``<NeMo_git_root>/examples/asr/conf/fastconformer/hybrid_transducer_ctc/``
``<NeMo_git_root>/examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/``

Note Hybrid models are being exported as RNNT (encoder and decoder+joint parts) by default.
To export as CTC (single encoder+decoder graph), `model.set_export_config({'decoder_type' : 'ctc'})` should be called before export.
Or, if ``<NeMo_git_root>/scripts/export.py`` is being used:
`python export.py hybrid_transducer.nemo hybrid_transducer.onnx --export-config decoder_type=ctc`

.. _Hybrid-ASR-TTS_model:

Hybrid ASR-TTS Model
--------------------

Hybrid ASR-TTS Model (``ASRWithTTSModel``) is a transparent wrapper for the ASR model with a frozen pretrained text-to-spectrogram model. The approach is described in the paper
`Text-only domain adaptation for end-to-end ASR using integrated text-to-mel-spectrogram generator <https://arxiv.org/abs/2302.14036>`_.
This allows using text-only data for training and finetuning, mixing it with audio-text pairs if necessary.

The model consists of three models:

* ASR model (``EncDecCTCModelBPE`` or ``EncDecRNNTBPEModel``)
* Frozen TTS Mel Spectrogram Generator (currently, only :ref:`FastPitch <FastPitch_model>` model is supported)
* Optional frozen :ref:`Spectrogram Enhancer model <SpectrogramEnhancer_model>` model trained to mitigate mismatch between real and generated mel spectrogram

    .. image:: images/hybrid_asr_tts_model.png
        :align: center
        :alt: Hybrid ASR-TTS Model
        :scale: 50%

For the detailed information see:

* :ref:`Text-only dataset <Hybrid-ASR-TTS_model__Text-Only-Data>` preparation
* :ref:`Configs and training <Hybrid-ASR-TTS_model__Config>`


.. _Confidence-Ensembles:

Confidence-based Ensembles
--------------------------

Confidence-based ensemble is a simple way to combine multiple models into a single system by only retaining the
output of the most confident model. Below is a schematic illustration of how such ensembles work.

    .. image:: images/conf-ensembles-overview.png
        :align: center
        :alt: confidence-based ensembles
        :scale: 50%

For more details about this model, see the `paper <https://arxiv.org/abs/2306.15824>`_
or read our `tutorial <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Confidence_Ensembles.ipynb>`_.

NeMo support Confidence-based Ensembles through the
:ref:`nemo.collections.asr.models.confidence_ensemble.ConfidenceEnsembleModel <confidence-ensembles-api>` class.

A typical workflow to create and use the ensemble is like this

1. Run `scripts/confidence_ensembles/build_ensemble.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/confidence_ensembles/build_ensemble.py>`_
   script to create ensemble from existing models. See the documentation inside the script for usage examples
   and description of all the supported functionality.
2. The script outputs a checkpoint that combines all the models in an ensemble. It can be directly used to transcribe
   speech by calling ``.trascribe()`` method or using
   `examples/asr/transcribe_speech.py <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py>`_.

Note that the ensemble cannot be modified after construction (e.g. it does not support finetuning) and only
transcribe functionality is supported (e.g., ``.forward()`` is not properly defined).

.. _Jasper_model:

Jasper
------

Jasper ("Just Another Speech Recognizer") :cite:`asr-models-li2019jasper` is a deep time delay neural network (TDNN) comprising of
blocks of 1D-convolutional layers. The Jasper family of models are denoted as ``Jasper_[BxR]`` where ``B`` is the number of blocks
and ``R`` is the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D convolution, batch normalization,
ReLU, and dropout:

    .. image:: images/jasper_vertical.png
        :align: center
        :alt: jasper model
        :scale: 50%

Jasper models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModel` class.

.. _Quartznet_model:

QuartzNet
---------

QuartzNet :cite:`asr-models-kriman2019quartznet` is a version of Jasper :cite:`asr-models-li2019jasper` model with separable
convolutions and larger filters. It can achieve performance similar to Jasper but with an order of magnitude fewer parameters.
Similarly to Jasper, the QuartzNet family of models are denoted as ``QuartzNet_[BxR]`` where ``B`` is the number of blocks and ``R``
is the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch normalization,
ReLU, and dropout:

    .. image:: images/quartz_vertical.png
        :align: center
        :alt: quartznet model
        :scale: 40%

QuartzNet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModel` class.


.. _Citrinet_model:

Citrinet
--------

Citrinet is a version of QuartzNet :cite:`asr-models-kriman2019quartznet` that extends ContextNet :cite:`asr-models-han2020contextnet`,
utilizing subword encoding (via Word Piece tokenization) and Squeeze-and-Excitation mechanism :cite:`asr-models-hu2018squeeze` to
obtain highly accurate audio transcripts while utilizing a non-autoregressive CTC based decoding scheme for efficient inference.

    .. image:: images/citrinet_vertical.png
        :align: center
        :alt: citrinet model
        :scale: 50%

Citrinet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModelBPE` class.

.. _ContextNet_model:

ContextNet
----------

ContextNet is a model uses Transducer/RNNT loss/decoder and is introduced in :cite:`asr-models-han2020contextnet`.
It uses Squeeze-and-Excitation mechanism :cite:`asr-models-hu2018squeeze` to model larger context.
Unlike Citrinet, it has an autoregressive decoding scheme.

ContextNet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecRNNTBPEModel` class for a
model with sub-word encoding and :class:`~nemo.collections.asr.models.EncDecRNNTModel` for char-based encoding.

You may find the example config files of ContextNet model with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/contextnet_rnnt/contextnet_rnnt_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/contextnet_rnnt/contextnet_rnnt.yaml``.

.. _Squeezeformer-CTC_model:

Squeezeformer-CTC
-----------------

Squeezeformer-CTC is a CTC-based variant of the Squeezeformer model introduced in :cite:`asr-models-kim2022squeezeformer`. Squeezeformer-CTC has a
similar encoder as the original Squeezeformer but uses CTC loss and decoding instead of RNNT/Transducer loss, which makes it a non-autoregressive model. The vast majority of the architecture is similar to Conformer model, so please refer to `Conformer-CTC <./models.html#conformer-ctc>`_.

The model primarily differs from Conformer in the following ways :

* Temporal U-Net style time reduction, effectively reducing memory consumption and FLOPs for execution.
* Unified activations throughout the model.
* Simplification of module structure, removal of redundant layers.

Here is the overall architecture of the encoder of Squeezeformer-CTC:

    .. image:: images/squeezeformer.png
        :align: center
        :alt: Squeezeformer-CTC Model
        :scale: 50%

This model supports both the sub-word level and character level encodings. You can find more details on the config files for the
Squeezeformer-CTC models at `Squeezeformer-CTC <./configs.html#squeezeformer-ctc>`_. The variant with sub-word encoding is a BPE-based model
which can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModelBPE` class, while the
character-based variant is based on :class:`~nemo.collections.asr.models.EncDecCTCModel`.

You may find the example config files of Squeezeformer-CTC model with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/squeezeformer/squeezeformer_ctc_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/squeezeformer/squeezeformer_ctc_bpe.yaml``.

.. _LSTM-Transducer_model:

LSTM-Transducer
---------------

LSTM-Transducer is a model which uses RNNs (eg. LSTM) in the encoder. The architecture of this model is followed from suggestions in :cite:`asr-models-he2019streaming`.
It uses RNNT/Transducer loss/decoder. The encoder consists of RNN layers (LSTM as default) with lower projection size to increase the efficiency.
Layer norm is added between the layers to stabilize the training.
It can be trained/used in unidirectional or bidirectional mode. The unidirectional mode is fully causal and can be used easily for simple and efficient streaming. However the accuracy of this model is generally lower than other models like Conformer and Citrinet.

This model supports both the sub-word level and character level encodings. You may find the example config file of RNNT model with wordpiece encoding at ``<NeMo_git_root>/examples/asr/conf/lstm/lstm_transducer_bpe.yaml``.
You can find more details on the config files for the RNNT models at `LSTM-Transducer <./configs.html#lstm-transducer>`_.

.. _LSTM-CTC_model:

LSTM-CTC
--------

LSTM-CTC model is a CTC-variant of the LSTM-Transducer model which uses CTC loss/decoding instead of Transducer.
You may find the example config file of LSTM-CTC model with wordpiece encoding at ``<NeMo_git_root>/examples/asr/conf/lstm/lstm_ctc_bpe.yaml``.


References
----------

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: ASR-MODELS
    :keyprefix: asr-models-
