# Audio Processing

The NeMo Audio collection provides models, datasets, and neural modules for speech enhancement, restoration,
dereverberation, and source extraction with single-channel or multichannel audio.

## Model families

* Mask-based models for single-channel processing and multichannel beamforming
* Predictive models using NCSN++, Spectrogram Conformer, or Spectrogram Conformer U-Net estimators
* Score-based, Schrödinger bridge, and flow matching generative models
* NVIDIA Maxine BNR 2.0 for removing background noise from single-channel speech

The API also provides guided source separation (GSS) and WPE dereverberation components for custom systems.

You can train from NeMo manifests, Lhotse CutSets, or Lhotse Shar datasets. The Lhotse loader can select channels and add
noise, room impulse responses, bandwidth limits, codec artifacts, clipping, or saturation while loading each batch.
For speech restoration, a public flow matching checkpoint can be fine-tuned with paired data or with clean recordings
that Lhotse degrades during loading.

## Get started

* Read the [Audio collection documentation](https://docs.nvidia.com/nemo/speech/nightly/audio/intro.html), including
  [models](https://docs.nvidia.com/nemo/speech/nightly/audio/models.html),
  [checkpoints](https://docs.nvidia.com/nemo/speech/nightly/audio/checkpoints.html),
  [datasets](https://docs.nvidia.com/nemo/speech/nightly/audio/datasets.html),
  [training and fine-tuning](https://docs.nvidia.com/nemo/speech/nightly/audio/fine_tuning.html), and
  [inference and evaluation](https://docs.nvidia.com/nemo/speech/nightly/audio/inference.html).
* Browse the [training configurations](../../../examples/audio/conf) and run
  [Audio model training](../../../examples/audio/audio_to_audio_train.py).
* Process models that implement `AudioToAudioModel.process()` with
  [the inference example](../../../examples/audio/process_audio.py), and evaluate outputs with
  [the evaluation example](../../../examples/audio/audio_to_audio_eval.py). The Maxine BNR notebook below uses its
  model-specific interface.
* Work through the [Audio tutorials](../../../tutorials/audio/README.md).
