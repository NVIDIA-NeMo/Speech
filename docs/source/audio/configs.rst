Configuration Files
===================

The Audio collection uses ``examples/audio/audio_to_audio_train.py`` for training. Complete configurations live
under `examples/audio/conf <https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/audio/conf>`_; select one with
``--config-name=<filename without .yaml>``. If no name is supplied, the entry point uses ``masking.yaml``.

This page describes the parts of those configurations that are specific to Audio models. See :doc:`Models <./models>`
for the model families, :doc:`Datasets <./datasets>` for NeMo and Lhotse data blocks, and :doc:`Training and Fine-Tuning
<./fine_tuning>` for runnable commands. Shared trainer and experiment settings are documented under :doc:`PyTorch
Lightning Training </core/core>` and :doc:`Experiment Manager <../core/exp_manager>`.

Available Configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Configuration
     - Use it for
   * - ``masking.yaml``
     - Estimating a mask and applying it to a selected input channel.
   * - ``masking_with_online_augmentation.yaml``
     - Training a mask estimator while Lhotse adds noise and room impulse responses.
   * - ``beamforming.yaml``
     - Estimating masks and then beamforming multichannel audio.
   * - ``beamforming_flex_channels.yaml``
     - Training a spatial filter that accepts a variable number of input channels.
   * - ``predictive.yaml``
     - Predicting a spectrogram directly with an NCSN++ estimator.
   * - ``predictive_conformer.yaml``
     - Predicting the target directly with a Conformer estimator.
   * - ``predictive_conformer_unet.yaml``
     - Predicting the target directly with a skip-connected Conformer U-Net.
   * - ``streaming_predictive_conformer.yaml``
     - Predicting the target with a causal Conformer and limited attention context.
   * - ``streaming_predictive_conformer_unet.yaml``
     - Predicting the target with a causal Conformer U-Net and limited attention context.
   * - ``score_based_generative.yaml``
     - Training SGMSE+ with a score estimator, SDE, and predictor-corrector sampler.
   * - ``schroedinger_bridge.yaml``
     - Training a Schrödinger bridge with a noise schedule and a sampler for the reverse process.
   * - ``flow_matching_generative.yaml``
     - Training a conditional flow matching model with an ODE sampler.
   * - ``flow_matching_generative_ssl_pretraining.yaml``
     - Self-supervised pretraining by masking spectrogram patches in a Lhotse Shar dataset.
   * - ``flow_matching_generative_finetuning.yaml``
     - Fine-tuning a flow matching model on paired data or clean data that Lhotse augments during loading.
   * - ``maxine_bnr.yaml``
     - Removing background noise from single-channel, 16 kHz speech with Maxine BNR 2.0.

How Audio Configurations Are Organized
---------------------------------------

Start with the configuration for the closest model family. Replace a component only with one that implements the
interface and tensor shapes expected by that family. Within ``model``, a component's ``_target_`` names the class to
instantiate and the remaining fields configure that class. The processing blocks differ by family:

* Mask-based models use ``encoder``, ``mask_estimator``, ``mask_processor``, and ``decoder``. The processor applies
  the estimated mask or uses it to estimate a spatial filter. A recipe can also apply ``channel_augment`` to its
  training inputs.
* Predictive models use ``encoder``, ``estimator``, and ``decoder``. The estimator predicts the target representation
  directly.
* SGMSE+ configurations add ``sde`` and ``sampler`` to the encoder, estimator, and decoder. The SDE defines the
  diffusion process used for training, and the sampler runs its reverse process at inference time.
* Schrödinger bridge configurations use ``noise_schedule`` and ``sampler`` with the encoder, estimator, and decoder.
* Flow matching configurations use ``flow`` and ``sampler`` with the encoder, estimator, and decoder. The
  self-supervised pretraining recipe also defines ``ssl_pretrain_masking``.
* Maxine BNR uses its built-in SEASR network rather than separate encoder, estimator, and decoder blocks.

The ``loss`` block has a different role in each family. Mask-based and predictive models compare decoded audio with
the target audio. SGMSE+ and flow matching train the estimator in the encoded domain. A Schrödinger bridge can use one
encoded-domain ``loss``, or a weighted combination of ``loss_encoded`` and ``loss_time``. Maxine BNR uses a combined
loss with configurable SI-SNR, spectral, and optional ASR loss weights.

Metrics under ``model.metrics.val`` and ``model.metrics.test`` compare decoded output with target audio. NeMo wraps
each configured TorchMetrics metric so it can ignore padding beyond each example's valid length and, when ``channel``
is set, evaluate one output channel. For a metric that has not been verified by the collection, set
``metric_using_batch_averaging: true`` after confirming that it averages over the batch. Validation and test loss are
included automatically and must not be added to these blocks.

``model.normalize_input`` peak-normalizes each input inside predictive and generative models, then restores the output
to the input scale. During generative training, the same input-derived scale is applied to the target before the loss
is computed. The NeMo manifest dataset option ``normalization_signal`` instead chooses which loaded signal supplies a
scale that is applied to every signal in that example.

``model.skip_nan_grad: true`` checks for non-finite gradients across all distributed ranks and clears the gradients on
every rank when any rank detects one.

Changing a Configuration
------------------------

Several Audio settings describe the same signal representation and must remain consistent when a configuration is
changed:

* A dataset inherits ``model.sample_rate`` when its own ``sample_rate`` is absent or ``null``. An explicit different
  value is kept and produces a warning.
* Analysis and synthesis parameters must match. Keep ``fft_length``, ``hop_length``, ``magnitude_power``, and ``scale``
  identical. When a configuration interpolates these values from ``model.encoder`` into ``model.decoder``, preserve the
  interpolations.
* Estimator input and output channels must match the tensors assembled by the model and ``model.num_outputs``. If an
  estimator has a frequency dimension, it must match the encoder output; an STFT produces
  ``fft_length // 2 + 1`` frequency bins.
* A Schrödinger bridge configuration must define ``loss``, or at least one of ``loss_encoded`` and ``loss_time``. Do
  not combine ``loss`` with an encoded-domain or time-domain loss.
* Flow matching configurations set the training target in ``model.estimator_target`` and tell the sampler how to
  interpret the output in ``model.sampler.estimator_target``. Set both to the same value:
  ``conditional_vector_field`` or ``data``.

The default ``masking.yaml`` contains ``train_ds``, ``validation_ds``, and ``test_ds`` blocks with required manifest
paths. Supply all three paths, or remove ``model.test_ds`` if the script should not run tests after training. See
:doc:`Datasets <./datasets>` for the Audio signal fields and :doc:`Lhotse Dataloading </dataloaders>` for shared loader
and augmentation options.
