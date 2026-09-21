NeMo Audio API
==============

Use this reference when building a custom Audio model or calling datasets and modules directly. To choose and run a
supplied configuration or checkpoint, start with :doc:`Models <./models>` and :doc:`Inference and Evaluation
<./inference>`.

Model Classes
-------------
Base Classes
~~~~~~~~~~~~
.. autoclass:: nemo.collections.audio.models.AudioToAudioModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Processing Models
~~~~~~~~~~~~~~~~~
.. autoclass:: nemo.collections.audio.models.EncMaskDecAudioToAudioModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.audio.models.FlowMatchingAudioToAudioModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.audio.models.PredictiveAudioToAudioModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.audio.models.ScoreBasedGenerativeAudioToAudioModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start

.. autoclass:: nemo.collections.audio.models.SchroedingerBridgeAudioToAudioModel
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Background Noise Removal
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nemo.collections.audio.models.maxine.bnr.BNR2
    :show-inheritance:
    :members:
    :exclude-members: setup_training_data, setup_validation_data, training_step, on_validation_epoch_end, validation_step, setup_test_data, on_train_epoch_start


Modules
-------

Features
~~~~~~~~
.. autoclass:: nemo.collections.audio.modules.features.SpectrogramToMultichannelFeatures
    :show-inheritance:
    :members:


Conformer Estimators
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nemo.collections.audio.parts.submodules.conformer.SpectrogramConformer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.conformer_unet.ConformerEncoderUNet
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.conformer_unet.SpectrogramConformerUNet
    :show-inheritance:
    :members:


Masking
~~~~~~~
.. autoclass:: nemo.collections.audio.modules.masking.MaskEstimatorRNN
    :show-inheritance:
    :members:

.. _audio-api-masking-multi-channel-mask-estimator:
.. autoclass:: nemo.collections.audio.modules.masking.MaskEstimatorFlexChannels
    :show-inheritance:
    :members:

.. _audio-api-masking-guided-source-separation:
.. autoclass:: nemo.collections.audio.modules.masking.MaskEstimatorGSS
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.modules.masking.MaskReferenceChannel
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.modules.masking.MaskBasedBeamformer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.modules.masking.MaskBasedDereverbWPE
    :show-inheritance:
    :members:


Projections
~~~~~~~~~~~

.. autoclass:: nemo.collections.audio.modules.projections.MixtureConsistencyProjection
    :show-inheritance:
    :members:


Self-Supervised Pretraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.audio.modules.ssl_pretrain_masking.SSLPretrainWithMaskedPatch
    :show-inheritance:
    :members:


Transforms
~~~~~~~~~~

.. _audio-api-audio-to-spectrogram:
.. autoclass:: nemo.collections.audio.modules.transforms.AudioToSpectrogram
    :show-inheritance:
    :members:

.. _audio-api-spectrogram-to-audio:
.. autoclass:: nemo.collections.audio.modules.transforms.SpectrogramToAudio
    :show-inheritance:
    :members:


Diffusion
---------
.. autoclass:: nemo.collections.audio.parts.submodules.diffusion.StochasticDifferentialEquation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.diffusion.OrnsteinUhlenbeckVarianceExplodingSDE
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.diffusion.ReverseStochasticDifferentialEquation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.diffusion.PredictorCorrectorSampler
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.diffusion.Predictor
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.diffusion.ReverseDiffusionPredictor
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.diffusion.Corrector
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.diffusion.AnnealedLangevinDynamics
    :show-inheritance:
    :members:


Flow Matching
-------------
.. autoclass:: nemo.collections.audio.parts.submodules.flow.ConditionalFlow
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.flow.OptimalTransportFlow
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.flow.ConditionalFlowMatchingSampler
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.flow.ConditionalFlowMatchingEulerSampler
    :show-inheritance:
    :members:

Multichannel Processing
-----------------------

.. autoclass:: nemo.collections.audio.parts.submodules.multichannel.ChannelAugment
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.multichannel.TransformAverageConcatenate
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.multichannel.TransformAttendConcatenate
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.multichannel.ChannelAveragePool
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.multichannel.ChannelAttentionPool
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.multichannel.ParametricMultichannelWienerFilter
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.multichannel.ReferenceChannelEstimatorSNR
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.multichannel.WPEFilter
    :show-inheritance:
    :members:


NCSN++
------

.. autoclass:: nemo.collections.audio.parts.submodules.ncsnpp.SpectrogramNoiseConditionalScoreNetworkPlusPlus
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.ncsnpp.NoiseConditionalScoreNetworkPlusPlus
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.ncsnpp.GaussianFourierProjection
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.ncsnpp.ResnetBlockBigGANPlusPlus
    :show-inheritance:
    :members:


Schrödinger Bridge
--------------------

.. autoclass:: nemo.collections.audio.parts.submodules.schroedinger_bridge.SBNoiseSchedule
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.schroedinger_bridge.SBNoiseScheduleVE
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.schroedinger_bridge.SBNoiseScheduleVP
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.schroedinger_bridge.SBSampler
    :show-inheritance:
    :members:


Transformer U-Net
-----------------

.. autoclass:: nemo.collections.audio.parts.submodules.transformerunet.LearnedSinusoidalPosEmb
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.transformerunet.ConvPositionEmbed
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.transformerunet.RMSNorm
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.transformerunet.AdaptiveRMSNorm
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.transformerunet.GEGLU
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.transformerunet.TransformerUNet
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.parts.submodules.transformerunet.SpectrogramTransformerUNet
    :show-inheritance:
    :members:


Losses
------

.. autoclass:: nemo.collections.audio.losses.MAELoss
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.losses.MSELoss
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.losses.SDRLoss
    :show-inheritance:
    :members:


Metrics
-------

.. autoclass:: nemo.collections.audio.metrics.AudioMetricWrapper
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.metrics.SquimMOSMetric
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.metrics.SquimObjectiveMetric
    :show-inheritance:
    :members:


Datasets
--------

NeMo Manifest Datasets
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.audio.data.audio_to_audio.BaseAudioDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.data.audio_to_audio.AudioToTargetDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.data.audio_to_audio.AudioToTargetWithReferenceDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.data.audio_to_audio.AudioToTargetWithEmbeddingDataset
    :show-inheritance:
    :members:


Lhotse Dataset
~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.audio.data.audio_to_audio_lhotse.LhotseAudioToTargetDataset
    :show-inheritance:
    :members:


Data Simulation
---------------

.. autoclass:: nemo.collections.audio.data.data_simulation.ArrayGeometry
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.data.data_simulation.RIRCorpusGenerator
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.audio.data.data_simulation.RIRMixGenerator
    :show-inheritance:
    :members:


Callbacks
---------

.. autoclass:: nemo.collections.audio.parts.utils.callbacks.SpeechEnhancementLoggingCallback
    :show-inheritance:
    :members:
