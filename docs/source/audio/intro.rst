Audio Processing
================

The NeMo Audio collection provides models, data loaders, and neural modules for single-channel and multichannel speech
and audio processing. The supplied configurations cover mask-based, predictive, and generative models.

These guides are for developers and researchers who want to run a pretrained Audio model, train or fine-tune a model
from a supplied configuration, or build a system from the collection's modules. They assume familiarity with Python,
PyTorch, command-line tools, and YAML. For an introduction to NeMo models and configuration, start with :doc:`Key
Concepts <../starthere/key_concepts>`.

Quick Start
-----------

After installing PyTorch for your platform, install NeMo with the Audio dependencies:

.. code-block:: bash

   uv pip install 'nemo-toolkit[audio]'

Then load a pretrained model for 16 kHz speech denoising and process a noisy recording:

.. code-block:: python

    from nemo.collections.audio.models import AudioToAudioModel

    model = AudioToAudioModel.from_pretrained("nvidia/se_den_sb_16k_small")
    output_files = model.process(
        paths2audio_files=["noisy.wav"],
        output_dir="enhanced",
    )
    print(output_files[0])

This checkpoint removes background noise from mono speech. NeMo loads the recording at the model's 16 kHz sample rate
and writes the result to ``enhanced/noisy.wav``. For other installation methods, see :ref:`installation`. For directory
processing, manifest processing, channel selection, sampler overrides, and objective evaluation, see :doc:`Inference
and Evaluation <./inference>`.

The Python API is available from an installed package. Commands that invoke ``examples/audio``, ``scripts``, or
``tools`` require a source checkout prepared as described in :ref:`install-from-source`. Run them from the repository
root with ``uv run``, or activate the checkout's ``.venv`` first.

Where to Start
--------------

To process recordings with a pretrained model, start with :doc:`Checkpoints <./checkpoints>` and :doc:`Inference and
Evaluation <./inference>`. To train a model or adapt pretrained weights, use :doc:`Models <./models>` to select an
architecture, :doc:`Configuration Files <./configs>` to find a configuration, and :doc:`Training and Fine-Tuning
<./fine_tuning>` for the training entry point.

Training data can be supplied as NeMo manifests, Lhotse CutSets, or Lhotse Shar datasets. The Lhotse loader can also
generate degraded inputs from clean recordings during training. See :doc:`Datasets <./datasets>` for Audio formats and
signal roles, :doc:`Training and Fine-Tuning <./fine_tuning>` for the clean-data workflow, and :doc:`Lhotse Dataloading
</dataloaders>` for shared loader and augmentation options.

For multichannel beamforming and the scope of the streaming-oriented Conformer configurations, see :doc:`Models
<./models>`.

Contents
--------

.. toctree::
   :maxdepth: 1

   models
   checkpoints
   inference
   fine_tuning
   datasets
   configs
   api
   resources
