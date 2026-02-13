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

FSDP2 Configuration
^^^^^^^^^^^^^^^^^^^

For Fully Sharded Data Parallel training:

1. Set ``data_parallel`` to the number of GPUs you want to use for data parallelism
2. Set ``tensor_parallel`` to 1 (disabled)

FSDP2 shards the model parameters across GPUs, all-gathers them for forward/backward passes, and then de-allocates after computation. This allows training of larger models with limited GPU memory.
See `PyTorch FSDP2 <https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>`_ for more details.

Tensor Parallelism Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
