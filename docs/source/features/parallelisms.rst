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

AutomodelParallelStrategy (SpeechLM2)
-------------------------------------

For SpeechLM2 models that use NeMo Automodel (for example ``SALMAutomodel``), the backbone LLM can be
too large for a single GPU. NeMo provides
``nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy``, a Lightning strategy that
delegates device mesh creation to NeMo Automodel and supports FSDP2, Tensor Parallelism (TP),
Sequence Parallelism (SP), Context Parallelism (CP), Expert Parallelism (EP) for MoE models, and
Hybrid Sharded Data Parallelism (HSDP).

**When to use:** When training or fine-tuning SpeechLM2 models whose LLM backbone does not fit
in a single GPU's memory, or when you want to scale training to many GPUs more efficiently
than DDP allows. Use ``AutomodelParallelStrategy`` for ``SALMAutomodel`` and MoE LLM backbones such
as NVIDIA Nemotron Nano V3.

**Requirements:** Each model must implement a ``configure_model()`` method that defines how its
layers are sharded and parallelized. ``SALMAutomodel`` already implements this and receives the
Automodel device mesh during ``configure_model()``. You cannot simply switch an arbitrary model
from DDP to ``AutomodelParallelStrategy`` without providing this implementation.

Concepts
^^^^^^^^

**FSDP2 (Fully Sharded Data Parallelism):**
    Shards model parameters, gradients, and optimizer states across GPUs in the data-parallel
    dimension. Dramatically reduces per-GPU memory -- enabling training of models that would not
    fit with DDP. Controlled via ``dp_size``; when ``dp_size`` is ``null``, NeMo Automodel infers
    it from the world size and the other parallelism dimensions.

**Tensor Parallelism (TP):**
    Splits individual weight matrices across GPUs. For example, a large linear layer's weight
    is partitioned column-wise or row-wise so each GPU holds only a slice. Controlled via
    ``tp_size``. The model must define a TP sharding plan (which layers are split and how).
    Automodel-backed SpeechLM2 models use the Automodel plan for the backbone LLM.

**Sequence Parallelism (SP):**
    Distributes activation memory along the sequence dimension across the TP group.
    SP is typically enabled alongside TP and reduces activation memory further. Enable it with
    ``distributed_config.sequence_parallel: true``.

**Context Parallelism (CP):**
    Splits long-context sequence processing across GPUs in the context-parallel group. Controlled
    via ``cp_size``. For SpeechLM2 models, CP is intended for packed-sequence training where each
    utterance is handled as its own attention segment.

**Expert Parallelism (EP):**
    Routes MoE experts across GPUs for MoE LLM backbones. Controlled via ``ep_size``. EP reuses
    the FSDP2 data-parallel axis: dense layers are sharded via FSDP2, while MoE expert layers use
    all-to-all expert routing on the same ranks.

**Hybrid Sharded Data Parallelism (HSDP):**
    Adds replication groups around FSDP2 sharding. Controlled via ``dp_replicate_size``.

Configuration
^^^^^^^^^^^^^

To enable ``AutomodelParallelStrategy`` for Automodel-backed SpeechLM2 models, replace the DDP
strategy block in the trainer config. The configured sizes must be compatible with the total
number of GPUs (``devices * num_nodes``). Leave ``dp_size: null`` to let NeMo Automodel infer the
data-parallel size from the remaining dimensions. ``ep_size`` controls MoE expert routing on the
data-parallel axis rather than adding a separate data-parallel dimension.

In YAML (with Hydra):

.. code-block:: yaml

    trainer:
        devices: 8
        num_nodes: 1
        accelerator: gpu
        precision: bf16-true
        strategy:
            _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
            dp_size: null          # inferred from world_size / other dimensions
            dp_replicate_size: 1   # HSDP replication group size
            tp_size: 1             # tensor parallel size
            cp_size: 1             # context parallel size
            ep_size: 8             # expert parallel size for MoE models

            distributed_config:
                sequence_parallel: false

            activation_checkpointing_llm: false
            activation_checkpointing_perception: false

In Python:

.. code-block:: python

    from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy

    trainer = pl.Trainer(
        strategy=AutomodelParallelStrategy(
            dp_size=None,
            dp_replicate_size=1,
            tp_size=1,
            cp_size=1,
            ep_size=8,
        ),
        devices=8,
        accelerator="gpu",
        precision="bf16-true",
        use_distributed_sampler=False,
    )

.. note::

    When using ``AutomodelParallelStrategy``, set ``use_distributed_sampler=False`` in the trainer.
    NeMo's data modules handle distributed sampling internally.

Activation Checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^

``AutomodelParallelStrategy`` exposes two activation-checkpointing knobs that can be enabled
independently:

* ``activation_checkpointing_llm`` checkpoints LLM transformer blocks. This single switch covers
  both the standard FSDP2 path and the EP/MoE parallelizer path, so use it for MoE LLM backbones
  whether ``ep_size`` is 1 or larger.
* ``activation_checkpointing_perception`` checkpoints the speech perception encoder layers before
  FSDP2 sharding.

Both options default to ``false``. Enable them to reduce activation memory at the cost of extra
recomputation during backward:

.. code-block:: yaml

    trainer:
        strategy:
            _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
            activation_checkpointing_llm: true
            activation_checkpointing_perception: true

Example: SALMAutomodel with FSDP2 only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest ``AutomodelParallelStrategy`` setup uses FSDP2 alone. This works when individual
layers fit in GPU memory:

.. code-block:: yaml

    trainer:
        devices: 8
        strategy:
            _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
            dp_size: 8
            tp_size: 1
            ep_size: 1

Example: SALMAutomodel with MoE Expert Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For MoE LLM backbones such as NVIDIA Nemotron Nano V3, use EP to distribute experts across GPUs.
Here, the dense layers use FSDP2 and MoE layers use 8-way expert routing:

.. code-block:: yaml

    trainer:
        devices: 8
        strategy:
            _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
            dp_size: null
            tp_size: 1
            ep_size: 8

Example: SALMAutomodel with TP + FSDP2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For larger dense LLM backbones, combine TP with FSDP2. Here, 2-way TP splits each layer across
2 GPUs and NeMo Automodel infers the FSDP2 data-parallel size from the remaining ranks:

.. code-block:: yaml

    trainer:
        devices: 8
        strategy:
            _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
            dp_size: null
            tp_size: 2
            ep_size: 1

ModelParallelStrategy (SALM and Duplex)
---------------------------------------

The original SpeechLM2 ``SALM`` and Duplex model configs use PyTorch Lightning's
``ModelParallelStrategy`` directly. This path is separate from ``SALMAutomodel`` and supports
FSDP2, TP, and SP using PyTorch-native DTensor.

**When to use:** Use ``ModelParallelStrategy`` for non-Automodel SpeechLM2 models, such as
``SALM`` and Duplex models. Use ``AutomodelParallelStrategy`` only for Automodel-backed models such
as ``SALMAutomodel``.

**Requirements:** As with ``AutomodelParallelStrategy``, the model must implement
``configure_model()`` to define how layers are sharded and parallelized. The SpeechLM2 SALM and
Duplex models already implement this.

ModelParallelStrategy Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The product of ``data_parallel_size`` and ``tensor_parallel_size`` must equal the total number of
GPUs (``devices * num_nodes``).

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

See the SpeechLM2 example configs in ``examples/speechlm2/conf/`` for complete training
configurations including data and optimizer settings.
