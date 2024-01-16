Common Configuration Files
============================

This section describes the NeMo configuration file setup that is specific to models in the MM NeRF collection. For general information
about how to set up and run experiments that is common to all NeMo models (e.g. Experiment Manager and PyTorch Lightning trainer
parameters), see the `Core Documentation <../../core/core.html>`_ section.

The model section of the NeMo Multimodal NeRF configuration files generally requires information about the dataset,
the background and/or foreground NeRF networks, renderer and the guidance model being used. The sections on
this page cover each of these in more detail.

Example configuration files for all of the NeMo Multimodal NeRF scripts can be found in the
config directory of the examples ``{NEMO_ROOT/examples/multimodal/generative/nerf/conf}``.


Trainer Configuration
---------------------

Trainer configuration specifies the arguments for Pytorch Lightning Trainer Object.

.. code-block:: yaml

  trainer:
    devices: 1                     # Number of GPUs for distributed, or the list of the GPUs to use e.g. [0, 1]
    num_nodes: 1                   # Number of nodes for distributed training
    precision: 16                  # Use 16 to enable or 32 for FP32 precision
    max_steps: 10000               # Number of training steps to perform
    accelerator: gpu               # accelerator to use, only "gpu" is officially supported
    enable_checkpointing: False    # Provided by exp_manager
    logger: False                  # Provided by exp_manager
    log_every_n_steps: 1           # Interval of logging
    val_check_interval: 100        # Interval of validation
    accumulate_grad_batches: 1     # Accumulates gradients over k batches before stepping the optimizer.
    benchmark: False               # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    enable_model_summary: True     # Enable or disable the model summarization


Refer to the `Pytorch Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html#>`__ API section
for all possible arguments


Experiment Manager Configurations
---------------------------------

NeMo Experiment Manager provides convenient way to configure logging, saving, resuming options and more.

.. code-block:: yaml

  exp_manager:
    name: ${name}                          # The name of the experiment.
    exp_dir: /results                      # Directory of the experiment, if None, defaults to "./nemo_experiments"
    create_tensorboard_logger: False       # Whether you want exp_manger to create a TensorBoard logger
    create_wandb_logger: False             # Whether you want exp_manger to create a Wandb logger
    wandb_logger_kwargs:                   # Wandb logger arguments
      project: dreamfusion
      group: nemo-df
      name: ${name}
      resume: True
    create_checkpoint_callback: True       # Whether you want Experiment manager to create a model checkpoint callback
    checkpoint_callback_params:            # Model checkpoint callback arguments
      every_n_epochs: 0
      every_n_train_steps:
      monitor: loss
      filename: '${name}-{step}'
      save_top_k: -1
      always_save_nemo: False
    resume_if_exists: True                 # Whether this experiment is resuming from a previous run
    resume_ignore_no_checkpoint: True      # Experiment manager errors out if resume_if_exists is True and no checkpoint could be found. This behavior can be disabled, in which case exp_manager will print a message and continue without restoring, by setting resume_ignore_no_checkpoint to True

Model Configuration
-------------------

Dataset Configuration
^^^^^^^^^^^^^^^^^^^^^

Training, validation, and test parameters are specified using the ``data`` sections in the model
configuration file. Depending on the task, there may be arguments specifying the augmentations
for the dataset, the image resolution, camera parameters and so on.

Any initialization parameter that is accepted for the Dataset class used in the experiment can be set in the config file.
Refer to the `Datasets <./datasets.html#Datasets>`__ section of the API for a list of Datasets and their respective parameters.

An example NeRF dataset configuration should look similar to the following:

.. code-block:: yaml

  model:
    data:
      train_batch_size: 1
      train_shuffle: false
      train_dataset:
        _target_: a pytorch Dataset or IterableDataset class

      val_batch_size: 1
      val_shuffle: false
      val_dataset:
        _target_: a pytorch Dataset or IterableDataset class

      test_batch_size: 1
      test_shuffle: false
      test_dataset:
        _target_: a pytorch Dataset or IterableDataset class


Model Architecture Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each configuration file should describe the model pipeline and architecture being used for the experiment.

Here is a list of modules a nerf pipeline might use:

+--------------------+-----------------------------------------------------+
| **Module**         | **Description**                                     |
+====================+=====================================================+
| :code:`guidance`   | guidance model                                      |
+--------------------+-----------------------------------------------------+
| :code:`nerf`       | the main network for foreground density and color   |
+--------------------+-----------------------------------------------------+
| :code:`background` | a complimentary layer for background color          |
+--------------------+-----------------------------------------------------+
| :code:`material`   | materials network for lightning and shading effects |
+--------------------+-----------------------------------------------------+
| :code:`renderer`   | rendering layer                                     |
+--------------------+-----------------------------------------------------+

Refer to `DreamFusion <./dreamfusion.html#dreamfusion>`_ for model specific configurations.


Optimizer Configurations
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  optim:
    name: adan
    lr: 5e-3
    eps: 1e-8
    weight_decay: 2e-5
    max_grad_norm: 5.0
    foreach: False


By default we use ``adan`` as the optimizer, refer to NeMo user guide for all supported optimizers.
