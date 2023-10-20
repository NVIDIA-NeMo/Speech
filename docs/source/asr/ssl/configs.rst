NeMo SSL Configuration Files
============================

This page covers NeMo configuration file setup that is specific to models in the Speech Self-Supervised Pre-training collection.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../../core/core`  page.

Dataset Configuration
---------------------

Dataset configuration for self-supervised model is mostly the same as for standard ASR training,
covered `here <../configs.html#Dataset Configuration>`__. The main difference is that in order to perform contrastive loss,
we will need to mask an equivalent amount of patches for all utterances in a batch. This means that we want to avoid
the durations varying too significantly within a single batch. There are several ways you can achieve this in NeMo:

1) The simplest way is to use the ``min_duration`` parameter in the dataset config, which will simply
discard all utterances below the specified length. This is a viable option if removing these utterances will not
significantly impact the total amount of hours of your dataset.

2) If your dataset contains many long utterances (longer than ~16 seconds) with varying length, then you may instead
want to use the ``random_segment`` perturbation, which will sample segments of a certain length from the full sample at
runtime (samples below the provided segment length will be padded). You can enable this by adding the following to your
dataset config:

.. code-block:: yaml

  augmentor:
    random_segment:
      prob: 1.0
      duration_sec: 16 # specify the duration you want

3) You can also use bucketing to ensure similar utterance lengths within batches.
See `Bucketing documentation <../datasets.html#bucketing-datasets>`__.

An example of SSL train and validation configuration should look similar to the following:

.. code-block:: yaml

  model:
    train_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      batch_size: 16 # you may increase batch_size if your memory allows
      shuffle: true
      num_workers: 8
      pin_memory: false
      use_start_end_token: true
      trim_silence: false
      max_duration: 16.7
      min_duration: 8.0
      # tarred datasets
      is_tarred: false
      tarred_audio_filepaths: null
      shuffle_n: 2048
      # bucketing params
      bucketing_strategy: "synced_randomized"
      bucketing_batch_size: null

    validation_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      batch_size: 16 # you may increase batch_size if your memory allows
      shuffle: false
      num_workers: 8
      pin_memory: true
      use_start_end_token: false
      min_duration: 8.0


Preprocessor Configuration
--------------------------

Preprocessor helps to compute MFCC or mel spectrogram features that are given as inputs to model.
For details on how to write this section, refer to `Preprocessor Configuration <../configs.html#preprocessor-configuration>`__

Augmentation Configurations
---------------------------

For self-supervised pre-training, we recommend using the ``MaskedPatchAugmentation`` class for spectrogram masking.
This augmentation divides utterances into fixed size patches, and then masks a fixed amount/fraction of them. You can
also add ``freq_masks`` and ``freq_width`` to apply masking to frequency bands.

If you are using contrastive loss with negatives sampled from masked steps in same utterance only,
make sure that the total amount of masked steps in each utterance will be big enough for the number of sampled negatives.
For example, if you are using 4x stride and want to sample 100 negatives, then you will need more than 400 masked steps.
If you are using the default ``patch_size`` of 48, then this means you will need to set ``mask_patches`` to at least 9.
When using a fraction of the total amount of patches instead of a fixed amount, you will need to make sure that the
minimum duration of your samples in large enough for the number of negatives to sample.

.. code-block:: yaml

  spec_augment:
    _target_: nemo.collections.asr.modules.MaskedPatchAugmentation
    patch_size: 48 # size of a single patch
    mask_patches: 0.5 # fraction of patches to mask (can be fixed int amount instead)
    freq_masks: 3 # Cut three frequency bands
    freq_width: 20 # ... of width 20 at maximum


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment. For self-supervised pre-training,
we will typically train the encoder of the model and then re-use it for fine-tuning, so the encoder can be configured in the same way
as you would for an ASR model. Note that any ASR model encoder can be used with any of the available pre-training methods,
though, given the same model sizes, we find the best downstream results when using `Conformer <./models.html#Conformer-Transducer>`__.

Unlike the encoders, the decoders and corresponding losses will be specific to the self-supervised pre-training, and are small enough that
you can discard them when transferring the model to fine-tuning.

The most basic method of pre-training we can use is to have the model solve a contrastive task
(this is the approach used in wav2vec 2.0 :cite:`ssl-models-wav2vec2`)
We can define the corresponding decoder and loss configs in the following way for an encoder with stride 4x.

.. code-block:: yaml

  decoder_out: 128

  decoder:
    _target_: nemo.collections.asr.modules.ConvASRDecoderReconstruction
    feat_in: ${model.encoder.d_model}
    feat_hidden: 128
    feat_out: ${model.decoder_out}
    stride_layers: 0
    # if loss.combine_time_steps is less than the encoder stride, then a corresponding amount of stride_layers needs to
    # be added to the decoder (here stride and combine_time_steps are both 4)
    non_stride_layers: 0

  loss:
    _target_: nemo.collections.asr.losses.ContrastiveLoss
    in_dim: ${model.preprocessor.features}
    proj_dim: ${model.decoder_out}
    combine_time_steps: 4 # how many spectrogram time steps are used for one target/representation for contrastive task
    quantized_targets: true # should quantizer or linear layer be used
    codebook_size: 300 # size of a single codebook for quantizer
    num_groups: 2 # number of codebooks to use for quantizer
    num_negatives: 100 # number of sampled negatives for each target
    sample_from_same_utterance_only: true # should negatives be sampled only from the same utterance
    sample_from_non_masked: false # should negatives be sampled from non-masked steps

Note that in the above example we combine 4 steps from the input spectrogram into a single "token" for the loss,
which corresponds to the encoder stride 4x. We might want to use different values for "combine_time_steps" and encoder stride.
In that case, we will need to add stride layers to decoders to match the strides. We can use the following example config
for a Citrinet encoder with stride 8x. In order to go from stride 8x to 4x, we use a single ``stride_layer`` in the decoder
with ``stride_transpose`` set to True.

.. code-block:: yaml

  decoder:
    _target_: nemo.collections.asr.modules.ConvASRDecoderReconstruction
    feat_in: ${model.model_defaults.enc_final}
    feat_hidden: 128
    feat_out: ${model.model_defaults.decoder_out_channels}
    stride_layers: 1
    #if loss.combine_time_steps is less than the encoder stride, then a corresponding amount of stride_layers needs to
    #be added to the decoder (here stride is 8 and combine_time_steps is 4, so 1 stride layer is added)
    non_stride_layers: 0
    stride_tranpose: true # whether to use transposed convolution for stride layers or not

  loss:
    _target_: nemo.collections.asr.losses.ContrastiveLoss
    in_dim: *n_mels
    proj_dim: ${model.model_defaults.decoder_out_channels}
    combine_time_steps: 4 #how many spectrogram time steps are used for one target/representation for contrastive task
    quantized_targets: false #should quantizer or linear layer be used
    sample_from_same_utterance_only: true #should negatives be sampled only from the same utterance
    sample_from_non_masked: false #should negatives be sampled from non-masked steps


It can be beneficial to combine contrastive loss with other losses, such as a masked language modeling (mlm) loss
(similar approach to W2V-Bert :cite:`ssl-models-w2v_bert`).
In order to do this, instead of specifying a single ``decoder`` and ``loss`` in the config, we can specify a ``loss_list``,
which can contain any amount of corresponding decoders and losses. For each decoder-loss pair,
we can specify a separate named sub-config, which contains the following fields:

1. ``decoder`` - The decoder config, specifying a ``target`` class and parameters.
2. ``loss`` - The corresponding loss config, specifying a ``target`` class and parameters.
3. ``loss_alpha`` - A multiplier on this loss (1.0 by default).
4. ``targets_from_loss`` - This parameter specifies which contrastive loss we should extract labels from. It is necessary for any loss which requires labels, if labels aren't present in your manifest.
5. ``transpose_encoded`` - This parameter is used to optionally transpose the encoded features before passing them into this loss.
6. ``start_step`` - The training step at which we should start using this decoder+loss.
7. ``output_from_layer`` - This parameter can be used to specify the name of the layer that we should extract encoded features from to pass into this decoder. If it's not specified or set to null, the final encoder layer is used.


The following is an example of a `loss_list` for a combination of contrastive+mlm losses,
where the mlm loss uses targets from the quantization module of the contrastive loss.


.. code-block:: yaml

  decoder_out: 128

  loss_list:
    contrastive:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoderReconstruction
        feat_in: ${model.encoder.d_model}
        feat_hidden: 128
        # features in hidden layer of decoder
        feat_out: ${model.decoder_out}
        stride_layers: 0
        # if loss.combine_time_steps is less than the encoder stride, then a corresponding amount of stride_layers needs to
        # be added to the decoder (here stride and combine_time_steps are both 4)
        non_stride_layers: 0
      loss:
        _target_: nemo.collections.asr.losses.ContrastiveLoss
        in_dim: ${model.preprocessor.features}
        proj_dim: ${model.decoder_out}
        combine_time_steps: 4 # how many spectrogram time steps are used for one target/representation for contrastive task
        quantized_targets: true # should quantizer or linear layer be used
        # (quantizer is required to extract pseudo-labels for other losses)
        codebook_size: 300
        num_groups: 2
        sample_from_same_utterance_only: true # should negatives be sampled only from the same utterance
        sample_from_non_masked: false # should negatives be sampled from non-masked steps

    mlm:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoder
        feat_in: ${model.encoder.d_model}
        num_classes: 90000
        # set this to be equal to codebook_size^groups in the contrastive loss
      loss:
        _target_: nemo.collections.asr.losses.MLMLoss
        combine_time_steps: 4
      targets_from_loss: "contrastive"
      # since this loss requires targets, we can either get them from a manifest or from a quantized contrastive loss
      loss_alpha: 1000.
      # multiplier applied to this loss relative to others
      transpose_encoded: false
      # transposing input may be necessary depending on which layer is used as input to decoder
      start_step: 0
      # determines what global step this loss starts being used at;
      # this can be set to a higher number if your training is long enough,
      # which may increase early training stability
      output_from_layer: null
      # if we wanted to use outputs from non-final encoder layer as input to this decoder,
      # the layer name should be specified here


We can also use other losses which require labels instead of mlm, such as ctc or rnnt loss. Since these losses, unlike mlm,
don't require our targets to have a direct alignment with our steps, we may also want to use set the ``reduce_ids`` parameter of the
contrastive loss to true, to convert any sequence of consecutive equivalent ids to a single occurrence of that id.

An example of a ``loss_list`` consisting of contrastive+ctc loss can look like this:

.. code-block:: yaml

  decoder_out: 128

  loss_list:
    contr:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoderReconstruction
        feat_in: ${model.encoder.d_model}
        feat_hidden: 128
        feat_out: ${model.decoder_out}
        stride_layers: 0
        non_stride_layers: 0
      loss:
        _target_: nemo.collections.asr.losses.ContrastiveLoss
        in_dim: ${model.preprocessor.features}
        proj_dim: ${model.decoder_out}
        combine_time_steps: 4
        quantized_targets: true
        codebook_size: 300
        num_groups: 2
        sample_from_same_utterance_only: true
        sample_from_non_masked: false
        reduce_ids: true

    ctc:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoder
        feat_in: ${model.encoder.d_model}
        num_classes: 90000
      loss:
        _target_: nemo.collections.asr.losses.CTCLossForSSL
        num_classes: 90000
      targets_from_loss: "contr"
      start_step: 3000

An example of contrastive+rnnt can look like this:

.. code-block:: yaml

  decoder_out: 128

  loss_list:
    contr:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoderReconstruction
        feat_in: ${model.encoder.d_model}
        feat_hidden: 128
        feat_out: ${model.decoder_out}
        stride_layers: 0
        non_stride_layers: 0
      loss:
        _target_: nemo.collections.asr.losses.ContrastiveLoss
        in_dim: ${model.preprocessor.features}
        proj_dim: ${model.decoder_out}
        combine_time_steps: 4
        quantized_targets: true
        codebook_size: 24
        sample_from_same_utterance_only: true
        sample_from_non_masked: false
        reduce_ids: true

    rnnt:
      decoder:
        _target_: nemo.collections.asr.modules.RNNTDecoderJointSSL
        decoder:
          _target_: nemo.collections.asr.modules.RNNTDecoder
          normalization_mode: null # Currently only null is supported for export.
          random_state_sampling: false # Random state sampling: https://arxiv.org/pdf/1910.11455.pdf
          blank_as_pad: true # This flag must be set in order to support exporting of RNNT models + efficient inference.
          vocab_size: 576
          prednet:
            pred_hidden: 640
            pred_rnn_layers: 1
            t_max: null
            dropout: 0.1
        joint:
          _target_: nemo.collections.asr.modules.RNNTJoint
          log_softmax: null  # 'null' would set it automatically according to CPU/GPU device
          preserve_memory: false  # dramatically slows down training, but might preserve some memory
          experimental_fuse_loss_wer: false
          jointnet:
            encoder_hidden: 512
            pred_hidden: 640
            joint_hidden: 640
            activation: "relu"
            dropout: 0.1
          num_classes: 576
      loss:
        _target_: nemo.collections.asr.losses.RNNTLossForSSL
        num_classes: 576
      targets_from_loss: "contr"
      start_step: 1000


We can also use multiple losses, which use features from different intermediate layers of the encoder as input :cite:`ssl-models-ssl_inter`.
In the following config example, we use contrastive loss + three different mlm losses, which use encoder outputs
respectively from 6th, 12th and final layer.

.. code-block:: yaml

  decoder_out: 128

  loss_list:
    contr:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoderReconstruction
        feat_in: ${model.encoder.d_model}
        feat_hidden: 128
        feat_out: ${model.decoder_out}
        stride_layers: 0
        non_stride_layers: 0
      loss:
        _target_: nemo.collections.asr.losses.ContrastiveLoss
        in_dim: ${model.preprocessor.features}
        proj_dim: ${model.decoder_out}
        combine_time_steps: 4
        quantized_targets: true
        codebook_size: 300
        sample_from_same_utterance_only: true
        sample_from_non_masked: false
      loss_alpha: 5.

    mlm:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoder
        feat_in: ${model.encoder.d_model}
        num_classes: 90000
      loss:
        _target_: nemo.collections.asr.losses.MLMLoss
        combine_time_steps: 4
      targets_from_loss: "contr"
      loss_alpha: 1000.

    mlm_2:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoder
        feat_in: ${model.encoder.d_model}
        num_classes: 90000
      loss:
        _target_: nemo.collections.asr.losses.MLMLoss
        combine_time_steps: 4
      targets_from_loss: "contr"
      loss_alpha: 300.
      output_from_layer: "layers.5"
      transpose_encoded: true

    mlm_3:
      decoder:
        _target_: nemo.collections.asr.modules.ConvASRDecoder
        feat_in: ${model.encoder.d_model}
        num_classes: 90000
      loss:
        _target_: nemo.collections.asr.losses.MLMLoss
        combine_time_steps: 4
      targets_from_loss: "contr"
      loss_alpha: 300.
      output_from_layer: "layers.11"
      transpose_encoded: true

References
-----------

.. bibliography:: ../asr_all.bib
    :style: plain
    :labelprefix: SSL-MODELS
    :keyprefix: ssl-models-