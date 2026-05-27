Training and Scaling
====================

This page provides detailed information on training speechlm2 models, including setup requirements, running experiments at scale, debugging, and parallelism strategies.

Running Experiments
-------------------

The speechlm2 collection includes several scripts to facilitate running experiments, especially on SLURM-based clusters.

SLURM Job Submission
^^^^^^^^^^^^^^^^^^^^

For training on SLURM clusters, use the following workflow:

.. code-block:: bash

    # Submit 8 consecutive jobs with random seeds
    scripts/speechlm2/auto_launcher_with_seed.sh -n8 s2s_tinyllama_repro.sub

The ``auto_launcher_with_seed.sh`` script:

1. Generates a random seed for each submitted job
2. Leverages ``shard_seed="randomized"`` in Lhotse to ensure each data parallel rank is seeded differently
3. Ensures each tensor parallel rank is seeded identically

SLURM Submission Script
^^^^^^^^^^^^^^^^^^^^^^^

Example ``s2s_tinyllama_repro.sub`` script:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=s2s_training
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=8
    #SBATCH --gres=gpu:8
    #SBATCH --time=24:00:00
    #SBATCH --exclusive
    #SBATCH --output=s2s_tinyllama_repro_%j.out

    # Check that the global random seed base is provided
    if [ -z "$1" ]; then
      echo "Usage: $0 <global_random_seed_base>"
      exit 1
    fi
    SEED=${1}

    EXP_NAME="s2s_training"
    RESULTS_DIR="results/${EXP_NAME}"

    srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
      python -u examples/speechlm2/s2s_duplex_train.py \
      --config-path=/path/to/config/dir \
      --config-name=s2s_training.yaml \
      exp_manager.name=${EXP_NAME} \
      exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
      trainer.num_nodes=$SLURM_JOB_NUM_NODES \
      exp_manager.explicit_log_dir=${RESULTS_DIR} \
      data.train_ds.seed=$SEED \
      data.validation_ds.seed=$SEED


Configuration Files
^^^^^^^^^^^^^^^^^^^

The main configuration file (``s2s_training.yaml``) contains all model, training, and data parameters. See :doc:`configs` for more details. It's recommended to copy and modify this file rather than overriding options in the SLURM script to maintain versioning and configuration clarity.

Debugging
---------

Running Locally with torchrun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For local debugging and profiling, use ``torchrun``:

.. code-block:: bash

    # Run with 4 GPUs locally
    torchrun --nproc_per_node=4 examples/speechlm2/s2s_duplex_train.py \
      --config-path=/path/to/config/dir \
      --config-name=s2s_training.yaml

Scaling Strategies
------------------

The speechlm2 collection includes support for model parallelism to scale training to large models across multiple GPUs.

Model Parallel Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

The collection supports multiple parallelism strategies:

1. **Fully Sharded Data Parallel (FSDP2)**: Distributes model parameters across GPUs
2. **Tensor Parallelism (TP)**: Splits individual tensors across GPUs
3. **Sequence Parallelism (SP)**: Splits sequence processing across GPUs
4. **2D Parallelism**: Combination of FSDP2 with TP/SP

AutomodelParallelStrategy (SALMAutomodel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``SALMAutomodel``, the collection provides ``AutomodelParallelStrategy`` which
delegates device mesh creation and parallelism to NeMo Automodel. This strategy
supports FSDP2, TP, PP, CP, EP (MoE), and HSDP.

.. code-block:: yaml

    trainer:
      strategy:
        _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
        dp_size: null       # inferred from world_size / other dims
        dp_replicate_size: 1  # HSDP replication group size
        tp_size: 1
        pp_size: 1
        cp_size: 1
        ep_size: 8          # Expert parallelism for MoE models

        # Activation checkpointing — two independent knobs:
        activation_checkpointing_llm: false         # LLM transformer blocks
        activation_checkpointing_perception: false  # speech encoder layers

The model's ``configure_model()`` receives the device mesh and passes it to
Automodel's ``from_pretrained`` for memory-efficient loading (each GPU only
loads its own shard).

The speech encoder / perception module currently only supports FSDP2 (controlled via ``dp_size``).

Activation Checkpointing
""""""""""""""""""""""""

``AutomodelParallelStrategy`` exposes two independent activation-checkpointing knobs:

* ``activation_checkpointing_llm`` — single switch covering both the non-EP FSDP2
  path (forces ``FSDP2Config.activation_checkpointing=True``) and the EP/MoE
  parallelizer path (passed through as a separate runtime arg). Use this for
  MoE LLMs whether ``ep_size`` is 1 or larger.
* ``activation_checkpointing_perception`` — wraps each transformer layer in
  ``perception.encoder.layers`` (and the Conformer ``pre_encode`` front-end when
  it isn't a bare ``nn.Linear``) with ``checkpoint_wrapper`` *before* FSDP2
  sharding. Implemented in ``AudioPerceptionModule.set_activation_checkpointing``.

Both default to ``false``. Toggle them independently to trade compute for
memory at either end of the model. They are SALMAutomodel-specific knobs (the
HF Transformers SALM path uses HuggingFace's own gradient-checkpointing API).

.. note::
   Expert Parallelism (EP) reuses the FSDP2 data-parallel axis (``dp_size``).
   Dense layers are sharded via FSDP2, while MoE expert layers use EP for
   all-to-all expert routing — both operate on the same set of GPUs.
   Setting ``ep_size`` controls how many GPUs participate in expert routing;
   it does not add a separate dimension.

Training with MoE LLM Backbones
""""""""""""""""""""""""""""""""

SALMAutomodel enables efficient training of Speech LLMs with Mixture-of-Experts
backbones like `NVIDIA Nemotron Nano V3 <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>`_
(30B total parameters, 3B active). NeMo Automodel provides two key MoE
optimizations:

* **Grouped GEMM**: Fuses all expert computations within a single MoE layer into
  one batched matrix multiplication, maximizing GPU utilization and throughput.
* **DeepEP** (Deep Expert Parallelism): An efficient all-to-all communication
  primitive for routing tokens to experts across GPUs, significantly reducing the
  communication overhead of Expert Parallelism.

Example: training SALMAutomodel with Nemotron Nano V3 on 8 GPUs with EP=8:

.. code-block:: bash

    torchrun --nproc_per_node=8 examples/speechlm2/salm_train.py \
      --config-name=salm_automodel \
      model.pretrained_llm=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      trainer.strategy.ep_size=8

For distributed inference, launch with ``torchrun``:

.. code-block:: bash

    torchrun --nproc_per_node=8 examples/speechlm2/salm_eval.py \
      pretrained_name=path/to/checkpoint \
      inputs=path/to/manifest \
      ep_size=2

Configuration
^^^^^^^^^^^^^

To configure parallelism, modify the ``trainer.strategy`` section in your YAML config:

.. code-block:: yaml

    trainer:
      strategy:
        _target_: nemo.core.ModelParallelStrategy
        find_unused_parameters: False
        data_parallel: 1   # World size for data parallelism (FSDP2)
        tensor_parallel: 8  # World size for tensor parallelism
      devices: 8
      num_nodes: 1
      accelerator: gpu
      precision: bf16-true

The model's ``configure_model`` method automatically sets up the appropriate parallelization based on this configuration.

FSDP2 Configuration (HF Automodel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Fully Sharded Data Parallel training:

1. Set ``data_parallel`` to the number of GPUs you want to use for data parallelism
2. Set ``tensor_parallel`` to 1 (disabled)

FSDP2 shards the model parameters across GPUs, all-gathers them for forward/backward passes, and then de-allocates after computation. This allows training of larger models with limited GPU memory.
See `PyTorch FSDP2 <https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>`_ for more details.

Tensor Parallelism Configuration (HF Automodel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Tensor Parallelism:

1. Set ``tensor_parallel`` to the number of GPUs you want to use for tensor parallelism
2. Set ``data_parallel`` to 1 (or higher for 2D parallelism)

The ``parallelize_module`` function applies a parallelization plan to specific model components, like splitting attention heads or embedding dimensions across GPUs.
See `PyTorch TP <https://pytorch.org/docs/stable/distributed.tensor.parallel.html>`_ for more details.

Implementation Details
----------------------

The core implementation of model parallelism is in the ``configure_model`` method of the model classes. Key aspects include:

1. **Module Sharding**: Calling ``fully_shard`` on modules to distribute parameters across data parallel ranks
2. **Parallelization Plans**: Creating and applying plans that specify how different layers should be parallelized
3. **Model-Specific Adaptations**: Handling architectural differences between different LLMs

Advanced Usage
--------------

Script Customization
^^^^^^^^^^^^^^^^^^^^

When customizing the training scripts, keep these points in mind:

1. **Path Overrides**: Override paths in the YAML configuration files with your own, as needed
2. **W&B Keys**: Update Weights & Biases API keys in configuration files
3. **Batch Size Tuning**: Adjust batch size based on your GPU memory and model size
