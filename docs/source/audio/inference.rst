Inference and Evaluation
========================

This page covers loading Audio models, processing files, and evaluating the results against target recordings.
See :doc:`Checkpoints <./checkpoints>` for pretrained models and :doc:`Datasets <./datasets>` for manifest formats.

The examples on this page call :meth:`~nemo.collections.audio.models.AudioToAudioModel.process`. Maxine BNR 2.0 has a
different inference API; use the notebook listed under :doc:`Resources <./resources>`.

Python API
----------

Load a pretrained model and process a list of files:

.. code-block:: python

   import torch

   from nemo.collections.audio.models import AudioToAudioModel

   model = AudioToAudioModel.from_pretrained("nvidia/se_den_sb_16k_small")
   if torch.cuda.is_available():
       model = model.cuda()
   output_files = model.process(
       paths2audio_files=["noisy/one.wav", "noisy/two.wav"],
       output_dir="enhanced",
       batch_size=2,
   )

:meth:`~nemo.collections.audio.models.AudioToAudioModel.process` writes each result at the model's sample rate, removes
batch padding, and returns the output paths. Pass ``input_dir`` to reproduce an input directory tree below
``output_dir``.

The denoising and dereverberation checkpoints listed in this guide accept mono audio and load it at 16 kHz.

For multichannel files, ``input_channel_selector`` accepts one channel index, a list of channel indices, ``average``,
or ``None`` for all channels. The selected number of channels must be compatible with the model configuration.

Process a Directory
-------------------

``examples/audio/process_audio.py`` restores either a pretrained model or a local ``.nemo`` checkpoint and recursively
processes a directory:

.. code-block:: bash

   python examples/audio/process_audio.py \
       pretrained_name=nvidia/se_den_sb_16k_small \
       audio_dir=/path/to/noisy_audio \
       audio_type=wav \
       output_dir=/path/to/enhanced_audio \
       batch_size=8 \
       amp=true

For a local checkpoint, replace ``pretrained_name`` with ``model_path``:

.. code-block:: bash

   python examples/audio/process_audio.py \
       model_path=/path/to/model.nemo \
       audio_dir=/path/to/noisy_audio \
       audio_type=wav \
       output_dir=/path/to/enhanced_audio

If ``cuda`` is omitted, the script uses GPU 0 when CUDA is available and otherwise uses the CPU. Automatic mixed
precision is enabled only when both ``amp=true`` and CUDA are available. Existing output directories are protected by
default; set ``overwrite_output=true`` to allow output files to be overwritten.

Process a Manifest
------------------

For a NeMo JSON manifest, provide the manifest and the key containing input paths:

.. code-block:: bash

   python examples/audio/process_audio.py \
       model_path=/path/to/model.nemo \
       dataset_manifest=/path/to/test_manifest.json \
       input_key=noisy_filepath \
       output_dir=/path/to/enhanced_audio \
       output_filename=/path/to/enhanced_manifest.json

The output manifest retains every processed input record and adds ``processed_audio_filepath``. When ``max_utts`` is
set, both processing and the output manifest stop at that limit.

Change Sampler Settings
-----------------------

Generative models expose sampler settings such as the number of integration steps. Override them without modifying the
checkpoint:

.. code-block:: bash

   python examples/audio/process_audio.py \
       pretrained_name=nvidia/se_den_sb_16k_small \
       audio_dir=/path/to/noisy_audio \
       ++sampler.num_steps=20

Only attributes already defined by the restored model's sampler may be overridden.

Objective Evaluation
--------------------

``examples/audio/audio_to_audio_eval.py`` can process a paired manifest and then compare each output with its target.
Each input and target pair must be aligned and have the same duration after loading.

.. code-block:: bash

   python examples/audio/audio_to_audio_eval.py \
       pretrained_name=nvidia/se_den_sb_16k_small \
       dataset_manifest=/path/to/test_manifest.json \
       input_key=noisy_filepath \
       target_key=clean_filepath \
       output_dir=/path/to/enhanced_audio \
       'metrics=[sdr,sisdr,estoi,pesq]' \
       batch_size=8

The script wraps TorchMetrics implementations of SDR, SI-SDR, STOI, ESTOI, and PESQ. The ``audio`` installation extra
includes TorchMetrics, ``pystoi``, and PESQ except on x86_64 macOS. NeMo's SQUIM wrappers use TorchAudio's pretrained
SQUIM models, so SQUIM metrics require a TorchAudio build compatible with the installed PyTorch version. PESQ expects
16 kHz wideband audio in this script; SQUIM may download its TorchAudio bundles on first use.

To score files that have already been processed, ensure the manifest contains both processed and target paths and run:

.. code-block:: bash

   python examples/audio/audio_to_audio_eval.py \
       only_score_manifest=true \
       dataset_manifest=/path/to/enhanced_manifest.json \
       processed_key=processed_audio_filepath \
       target_key=clean_filepath \
       'metrics=[sdr,estoi]'
