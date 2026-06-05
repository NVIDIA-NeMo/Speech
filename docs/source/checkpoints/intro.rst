Checkpoints
===========

In this section, we present the checkpoint formats supported by NVIDIA NeMo.

NeMo Checkpoints (.nemo)
-------------------------

A ``.nemo`` checkpoint is a tar archive that bundles model configurations (YAML), model weights (``.ckpt``),
and other artifacts like tokenizer models or vocabulary files. This consolidated design streamlines
sharing, loading, tuning, evaluating, and inference.

Because ``.nemo`` files are standard tar archives, you can unpack them, inspect or modify their contents,
and repack them:

.. code-block:: bash

    # Unpack
    mkdir model_contents && tar xf model.nemo -C model_contents/

    # Inspect / edit files inside
    ls model_contents/

    # Repack
    cd model_contents && tar cf ../model_modified.nemo * && cd ..

This is useful for inspecting model configs, swapping tokenizer files, or modifying configuration
without reloading the model in Python.

``.nemo`` checkpoints are the primary format for ASR, TTS, and Audio pretrained models.

PyTorch Lightning Checkpoints (.ckpt)
--------------------------------------

During training, PyTorch Lightning saves ``.ckpt`` files that contain model weights, optimizer
states, and training metadata (epoch, step, scheduler state). These are used to resume training
from where it left off.

SafeTensors (.safetensors)
--------------------------

`SafeTensors <https://huggingface.co/docs/safetensors>`_ is a format for storing tensors that is
safe (no arbitrary code execution, unlike pickle-based formats), fast (supports zero-copy and
lazy loading of individual tensors), and widely adopted across the HuggingFace ecosystem.

SpeechLM2 models use ``.safetensors`` as their primary checkpoint format, following the HuggingFace
model conventions. SpeechLM2 models are saved and loaded via HuggingFace Hub integration
(``save_pretrained`` / ``from_pretrained``), and their weights are stored in ``.safetensors`` files.

.. note::

    SpeechLM2 models do not use the ``.nemo`` format for their own checkpoints. The ``.nemo`` format
    is only used in the SpeechLM2 collection to load pretrained ASR checkpoints that initialize
    the speech encoder component.

Distributed Checkpoints
-----------------------

When training with ``ModelParallelStrategy`` (FSDP2 / Tensor Parallelism), PyTorch Lightning
automatically saves **distributed checkpoints**. Instead of gathering all shards onto a single
process, each process saves its own shard to a directory. This is significantly faster and uses
less memory than consolidating into a single file.

Distributed checkpoints are saved as a directory containing:

- A ``.metadata`` file describing the tensor layout across shards
- Numbered ``.distcp`` files with per-rank weight shards

PyTorch Lightning handles loading distributed checkpoints transparently -- you resume training
with the same ``ckpt_path`` argument regardless of whether the checkpoint is a single file or a
sharded directory.

.. code-block:: python

    # Resuming from a distributed checkpoint works the same as a regular checkpoint
    trainer.fit(model, ckpt_path="path/to/distributed_checkpoint_dir")
