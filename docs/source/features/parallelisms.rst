.. _parallelisms:

Parallelisms
============

NeMo uses native PyTorch parallelism primitives for distributed training, enabling efficient multi-GPU and multi-node
model training for Speech AI workloads.

Data Parallelism
----------------

Distributed Data Parallelism (DDP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DDP is the default parallelism strategy in NeMo. It replicates the model across all GPUs and synchronizes
parameter gradients via all-reduce after each backward pass, keeping model replicas consistent.

DDP is suitable for most Speech AI training workloads and is automatically enabled when using multiple GPUs
with PyTorch Lightning.

Fully Sharded Data Parallelism (FSDP2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FSDP2 shards model parameters, gradients, and optimizer states across data-parallel GPUs, significantly
reducing per-GPU memory usage. This enables training larger models or using larger batch sizes than
would be possible with DDP alone.

NeMo integrates with PyTorch's native FSDP2 implementation. FSDP2 is recommended when model states
exceed the memory capacity of a single GPU.

Tensor Parallelism
------------------

Tensor Parallelism (TP) partitions individual layer parameters across GPUs using PyTorch's DTensor
abstraction. This reduces both model state memory and activation memory per GPU.

TP is useful for very large models where even a single layer's parameters may not fit in a single
GPU's memory.

Sequence Parallelism
--------------------

Sequence Parallelism (SP) distributes activation memory along the sequence dimension across GPUs.
This is particularly helpful for models processing long audio sequences, as it reduces per-GPU
activation memory requirements.

Configuration
-------------

Parallelism settings are configured through the PyTorch Lightning trainer and strategy configuration.
Refer to the collection-specific training guides for detailed configuration examples:

- :doc:`ASR Training <../asr/intro>`
- :doc:`TTS Training <../tts/intro>`
- :doc:`SpeechLM2 Training <../speechlm2/intro>`
