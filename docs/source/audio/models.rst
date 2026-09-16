Models
======

Audio models take one or more input channels and return one or more processed channels. Use a pretrained checkpoint
when one matches the task, or start from a supplied configuration to train on your own data. See :doc:`Checkpoints
<./checkpoints>` for pretrained models and :doc:`Configuration Files <./configs>` for the available configurations.

All Audio model classes derive from :class:`~nemo.collections.audio.models.AudioToAudioModel`. The sections below
explain what each family does and which configurations exercise it. The configuration files are available in
`examples/audio/conf <https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/audio/conf>`_.

Mask-Based Processing
---------------------

:class:`~nemo.collections.audio.models.EncMaskDecAudioToAudioModel` contains an encoder, mask estimator, optional mask
processor, and decoder. The encoder and decoder can be learned or fixed transforms, such as the :ref:`short-time
Fourier transform <audio-api-audio-to-spectrogram>` and :ref:`inverse STFT <audio-api-spectrogram-to-audio>`.

For single-channel enhancement, a neural estimator predicts a mask that is applied to the encoded mixture.

.. _audio-multichannel-processing:

Multichannel Masking and Spatial Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``beamforming.yaml`` configuration estimates speech and noise masks with ``MaskEstimatorRNN`` and applies
:class:`~nemo.collections.audio.modules.masking.MaskBasedBeamformer`. It accepts all channels from the input recording
and trains against one selected target channel.

The ``beamforming_flex_channels.yaml`` configuration uses
:class:`~nemo.collections.audio.modules.masking.MaskEstimatorFlexChannels` and channel augmentation, followed by a
parametric multichannel Wiener filter. The supplied configuration sets ``filter_beta: 0``, which gives the MVDR
solution, and selects a reference channel from the estimated signal-to-noise ratio
:cite:`audio-models-jukic2023flexible,audio-models-souden2010`.

The flexible-channel estimator does not require a fixed array geometry and can be trained with a varying number of
input channels :cite:`audio-models-yoshioka2021vararray`. See :doc:`Configuration Files <./configs>` for the supplied
configurations and :doc:`Datasets <./datasets>` for multichannel input and target selection.

The API also exposes :class:`~nemo.collections.audio.modules.masking.MaskEstimatorGSS` for guided source separation
:cite:`audio-models-ito2016directional,audio-models-boeddeker2018gss` and
:class:`~nemo.collections.audio.modules.masking.MaskBasedDereverbWPE` for weighted prediction error dereverberation
:cite:`audio-models-yoshioka2012wpe`. These are component APIs; ``examples/audio/conf`` does not currently include a
complete GSS or WPE training configuration.

Predictive Models
-----------------

:class:`~nemo.collections.audio.models.PredictiveAudioToAudioModel` replaces the mask estimator and processor with a
neural estimator that directly predicts the target latent representation. ``predictive.yaml`` uses NCSN++; the
Conformer configurations use
:class:`~nemo.collections.audio.parts.submodules.conformer.SpectrogramConformer` or
:class:`~nemo.collections.audio.parts.submodules.conformer_unet.SpectrogramConformerUNet` estimators based on the
Conformer architecture :cite:`audio-models-gulati2020conformer_audio`.

The ``streaming_predictive_conformer*.yaml`` configurations use causal convolution and limit the estimator's left and
right attention context. They disable normalization across the complete utterance because that would require future
samples. Despite their names, ``process_audio.py`` still reads complete files, and the model API does not expose the
estimator cache. An application that processes chunks must therefore integrate the estimator cache directly or overlap
adjacent chunks. It must also provide streaming STFT and inverse STFT processing.

Generative Processing
---------------------

SGMSE+
~~~~~~

:class:`~nemo.collections.audio.models.ScoreBasedGenerativeAudioToAudioModel` uses a score network, stochastic
differential equation, and predictor-corrector sampler. The ``score_based_generative.yaml`` configuration combines an NCSN++
estimator with an Ornstein-Uhlenbeck variance-exploding SDE to implement the speech enhancement and dereverberation
model described by Richter et al. :cite:`audio-models-richter2023sgmse`.

Schrödinger Bridge
~~~~~~~~~~~~~~~~~~

:class:`~nemo.collections.audio.models.SchroedingerBridgeAudioToAudioModel` uses a Schrödinger bridge to transform
degraded audio into the target signal :cite:`audio-models-jukic2024sb`.

Flow Matching
~~~~~~~~~~~~~

:class:`~nemo.collections.audio.models.FlowMatchingAudioToAudioModel` learns a vector field from noise to the target
representation while conditioning on the degraded input. During inference, an ODE sampler starts from Gaussian noise
and follows that field to generate the output :cite:`audio-models-lipman2023flow,audio-models-ku2025generative`. The
supplied configurations use a Transformer U-Net whose design follows Voicebox and AudioBox
:cite:`audio-models-le2023voicebox,audio-models-vyas2023audiobox`.

``flow_matching_generative_finetuning.yaml`` can start from the public
``nvidia/sr_ssl_flowmatching_16k_430m`` checkpoint or from a local run of
``flow_matching_generative_ssl_pretraining.yaml``. It accepts paired degraded and clean recordings, or clean
recordings that Lhotse degrades during loading. See :doc:`Training and Fine-Tuning <./fine_tuning>` for both workflows
and the model settings that must match the checkpoint.

Maxine BNR 2.0
--------------

:class:`~nemo.collections.audio.models.maxine.bnr.BNR2` uses the time-domain SEASR architecture to remove background
noise from single-channel, 16 kHz speech :cite:`audio-models-remane2024seasr`. Train it with ``maxine_bnr.yaml``. The
corresponding notebook is listed under :doc:`Resources <./resources>`.

References
----------

The references below provide the architectural and algorithmic background for the Audio models and components
described on this page. They do not define benchmark protocols for the collection.

.. bibliography:: audio_all.bib
   :style: plain
   :labelprefix: AUDIO-
   :keyprefix: audio-models-
