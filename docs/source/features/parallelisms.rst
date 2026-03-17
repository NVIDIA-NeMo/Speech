.. _parallelisms:

Parallelisms
============

NeMo uses native PyTorch parallelism primitives for distributed training, enabling efficient multi-GPU and multi-node
model training for Speech AI workloads.

DDP (all collections)
---------------------

Distributed Data Parallelism (DDP) is the default strategy for all NeMo collections (ASR, TTS, Audio, SpeechLM2).
It replicates the entire model on every GPU, runs each GPU on a different data shard, and synchronizes
parameter gradients via all-reduce after each backward pass.

**When to use:** DDP works well when the full model fits in a single GPU's memory.
This covers the vast majority of ASR, TTS, and Audio training workloads.

DDP is enabled by default in NeMo. You can configure it explicitly in YAML:

.. code-block:: yaml

    trainer:
        strategy:
            _target_: lightning.pytorch.strategies.DDPStrategy
            gradient_as_bucket_view: true
            find_unused_parameters: true

Or in Python:

.. code-block:: python

    from lightning.pytorch.strategies import DDPStrategy

    trainer = pl.Trainer(
        strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True),
        devices=8,
        accelerator="gpu",
    )

ModelParallelStrategy (SpeechLM2)
---------------------------------

For SpeechLM2 models (e.g. SALM / Canary-Qwen), the backbone LLM can be too large for a single GPU.
PyTorch Lightning's ``ModelParallelStrategy`` enables FSDP2, Tensor Parallelism (TP), and
Sequence Parallelism (SP) using PyTorch-native DTensor.

**When to use:** When training or fine-tuning SpeechLM2 models whose LLM backbone does not fit
in a single GPU's memory, or when you want to scale training to many GPUs more efficiently
than DDP allows.

**Requirements:** Each model must implement a ``configure_model()`` method that defines how its
layers are sharded (FSDP2) and parallelized (TP / SP). The SpeechLM2 models (SALM, DuplexEARTTS)
already implement this. You cannot simply switch an arbitrary model from DDP to
``ModelParallelStrategy`` without providing this implementation.

Concepts
^^^^^^^^

**FSDP2 (Fully Sharded Data Parallelism):**
    Shards model parameters, gradients, and optimizer states across GPUs in the data-parallel
    dimension. Dramatically reduces per-GPU memory -- enabling training of models that would not
    fit with DDP. Controlled via the ``data_parallel_size`` argument.

**Tensor Parallelism (TP):**
    Splits individual weight matrices across GPUs. For example, a large linear layer's weight
    is partitioned column-wise or row-wise so each GPU holds only a slice. Controlled via the
    ``tensor_parallel_size`` argument. The model must define a TP sharding plan (which layers
    are split and how). SpeechLM2 models automatically use the HuggingFace TP plan for the
    backbone LLM when available.

**Sequence Parallelism (SP):**
    Distributes activation memory along the sequence dimension across the TP group.
    SP is typically enabled alongside TP and reduces activation memory further.

Configuration
^^^^^^^^^^^^^

To enable ``ModelParallelStrategy`` for SpeechLM2, replace the DDP strategy block in the
trainer config. The product of ``data_parallel_size`` and ``tensor_parallel_size`` must equal
the total number of GPUs (``devices * num_nodes``).

In YAML (with Hydra):

.. code-block:: yaml

    trainer:
        devices: 8
        num_nodes: 1
        accelerator: gpu
        precision: bf16-true
        strategy:
            _target_: lightning.pytorch.strategies.ModelParallelStrategy
            data_parallel_size: 4   # FSDP2: shard across 4 GPUs
            tensor_parallel_size: 2  # TP: split layers across 2 GPUs

In Python:

.. code-block:: python

    from lightning.pytorch.strategies import ModelParallelStrategy

    trainer = pl.Trainer(
        strategy=ModelParallelStrategy(
            data_parallel_size=4,
            tensor_parallel_size=2,
        ),
        devices=8,
        accelerator="gpu",
        precision="bf16-true",
        use_distributed_sampler=False,
    )

.. note::

    When using ``ModelParallelStrategy``, set ``use_distributed_sampler=False`` in the trainer.
    NeMo's data modules handle distributed sampling internally.

Example: SALM with FSDP2 only (no TP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest ``ModelParallelStrategy`` setup uses FSDP2 alone. This requires no TP plan
and works when individual layers fit in GPU memory:

.. code-block:: yaml

    trainer:
        devices: 8
        strategy:
            _target_: lightning.pytorch.strategies.ModelParallelStrategy
            data_parallel_size: 8
            tensor_parallel_size: 1

Example: SALM with TP + FSDP2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For larger LLM backbones, combine TP with FSDP2. Here, 2-way TP splits each layer across
2 GPUs within a node, and 4-way FSDP2 shards the model across 4 such groups:

.. code-block:: yaml

    trainer:
        devices: 8
        strategy:
            _target_: lightning.pytorch.strategies.ModelParallelStrategy
            data_parallel_size: 4
            tensor_parallel_size: 2

See the SpeechLM2 example configs in ``examples/speechlm2/conf/`` for complete training
configurations including data and optimizer settings.
