# NeMo Audio Processing Tutorials

Start with **Speech Enhancement with NeMo** for the complete training and inference workflow. The training examples are
intended for a GPU runtime.

## Tutorials

* [Speech Enhancement with NeMo](speech_enhancement/Speech_Enhancement_with_NeMo.ipynb) covers data preparation,
  model training, evaluation, and inference.
* [Speech Enhancement with Online Augmentation](speech_enhancement/Speech_Enhancement_with_Online_Augmentation.ipynb)
  uses Lhotse to convolve clean speech with room impulse responses and mix in noise during training.
* [BNR Speech Enhancement with NeMo](speech_enhancement/BNR_Speech_enhancement_with_NeMo.ipynb) demonstrates the
  NVIDIA Maxine BNR 2.0 model for removing background noise from single-channel speech.

## Documentation and examples

See the [Audio collection documentation](https://docs.nvidia.com/nemo/speech/nightly/audio/intro.html) for current
model families, datasets, checkpoints, fine-tuning, and APIs. Repository examples include:

* [Train an Audio model](../../examples/audio/audio_to_audio_train.py)
* [Process audio](../../examples/audio/process_audio.py) and
  [evaluate enhanced output](../../examples/audio/audio_to_audio_eval.py)
* [Save augmented audio and a Lhotse CutSet](../../examples/audio/save_augmented.py)
* [Browse model configurations](../../examples/audio/conf)
