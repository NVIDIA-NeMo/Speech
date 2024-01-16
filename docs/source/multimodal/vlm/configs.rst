Common Configuration Files
==========================

This section provides a detailed overview of the NeMo configuration file setup specific to models within the NeMo Multimodal Language Model collection. For foundational knowledge about setting up and executing experiments common to all NeMo models, such as the Experiment Manager and PyTorch Lightning trainer parameters, refer to the :doc:`../core/core` section.

Within the configuration files of the NeMo Multimodal Language Model, details concerning dataset(s), augmentation, optimization parameters, and model architectural specifications are central. This page explores each of these aspects.

Discover exemplary configuration files for all NeMo Multimodal Language Model scripts in the `config directory of the examples <http://TODOURL>`_.

Dataset Configuration
=====================

The configuration file delineates parameters for training, validation, and testing datasets under the ``train``, ``validation``, and ``test`` sections, respectively. Depending on the specific task, the configuration might include parameters related to dataset augmentations, image resolution filtering, among others.

All initialization parameters supported by the Dataset class utilized in the experiment can be defined in the config file. For a comprehensive list of Datasets and their associated parameters, consult the `Datasets <../api.html#Datasets>`__ section of the API.

A representative training configuration appears as:

.. code-block:: yaml

  data:
      num_workers: 16 # Number of workers for the dataloader
      train:
        dataset_path: # Paths to wdinfo files specifying training datasets
          - dataset1.pkl
          - dataset2.pkl
      validation: # Paths to validation dataset files (pkl or tar)
        dataset_path:
          - dataset3.pkl
      webdataset:
        use_webdataset: True     # Enable the use of webdataset
        infinite_sampler: False  # Set to False for correct training resumption and accurate epoch counting
        local_root_path: ???     # Specify the directory where the dataset is located
        verbose: False           # Enable verbose debugging information if set to True

As outlined in `Datasets <./datasets.html>`_, the utilization of the webdataset format is imperative for the vision-language contrastive training pipeline. Due to the extensive data requirements, it's beneficial to follow the `Webdataset Multinode Training Guideline <https://github.com/webdataset/webdataset#multinode-training>`_ for recommended practices.


.. code-block:: yaml

  data:
    train:
      augmentations:
        resize_smallest_side: 256 # Adjust the image's smallest side to the specified dimension
        center_crop_h_w: 256, 256 # Perform center cropping
        horizontal_flip: False # Apply horizontal flip
      filterings:
        resolution:
          method: larger
          value: 256

This configuration uses the same data class as the text2image setup. Enabling the ``train.filterings`` and ``train.augmentations`` sections offers flexibility to filter specific images (and their text pairs) for particular requirements without the need to recreate the entire webdataset. The provided example demonstrates filtering to retain only images with resolutions exceeding 256x256 for training. To concatenate multiple webdatasets, simply list all corresponding wdinfo files under ``train.dataset_path``.





Trainer Configuration
---------------------

This section outlines arguments for the Pytorch Lightning Trainer Object.

.. code-block:: yaml

  trainer:
    devices: 1 # number of GPUs (0 for CPU), or list of the GPUs to use e.g. [0, 1]
    num_nodes: 1
    max_epochs: -1
    max_steps: 2500000 # precedence over max_epochs
    logger: False  # Provided by exp_manager
    precision: bf16 # Should be set to 16 for O1 and O2 to enable the AMP.
    accelerator: gpu
    log_every_n_steps: 5  # Interval of logging.
    resume_from_checkpoint: null # The path to a checkpoint file to continue the training, restores the whole state including the epoch, step, LR schedulers, apex, etc.
    num_sanity_val_steps: 10 # number of steps to perform validation steps for sanity check the validation process before starting the training, setting to 0 disables it
    enable_checkpointing: False # Provided by exp_manager
    accumulate_grad_batches: 1 # do not modify, grad acc is automatic for training megatron models
    gradient_clip_val: 1.0
    benchmark: False
    enable_model_summary: True

For a detailed list of arguments, refer to the `Pytorch Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html#>`__ API section.

Experiment Manager Configurations
---------------------------------

The NeMo Experiment Manager provides a streamlined approach to manage various tasks such as logging, saving, and resuming.

.. code-block:: yaml

  exp_manager:
    exp_dir: null  # exp_dir for your experiment, if None, defaults to "./nemo_experiments"
    name: ${name}
    create_wandb_logger: True
    wandb_logger_kwargs: # Whether you want exp_manger to create a Wandb logger
      name: training-session
      project: text2img
      group: nemo
      resume: True
    create_tensorboard_logger: True  # Whether you want exp_manger to create a tb logger
    create_checkpoint_callback: True  # Whether you want exp_manager to create a modelcheckpoint callback
    checkpoint_callback_params:
      monitor: reduced_train_loss
      save_top_k: 5
      every_n_epochs: 0 # Save checkpoint frequency.
      every_n_train_steps: 1000 # Mutually exclusive with every_n_epochs. It is recommended to set this if training on large-scale dataset.
      filename: '${name}--{reduced_train_loss:.2f}-{step}-{consumed_samples}'
    resume_if_exists: True
    resume_ignore_no_checkpoint: True
    resume_from_checkpoint: ${model.resume_from_checkpoint}
    ema:
      enable: True
      decay: 0.9999
      validate_original_weights: False
      every_n_steps: 1
      cpu_offload: False

Optimizer Configurations
-------------------------

.. code-block:: yaml

  optim:
    name: fused_adam
    lr: 0.0001
    eps: 1e-8
    betas: [ 0.9, 0.999 ]
    weight_decay: 0.01
    sched:
      name: WarmupPolicy
      warmup_steps: 10000
      warmup_ratio: null

The default optimizer used is ``fused_adam``. For details on all supported optimizers, refer to the NeMo user guide. The learning rate scheduler can be specified in the ``optim.sched`` section.

Model Configurations
--------------------

Each configuration file should detail the model architecture used for the experiment.

The parameters commonly shared across most multimodal language models include:

+--------------------------------------+--------------+---------------------------------------------------------------------------------------+
| **Parameter**                        | **Datatype** | **Description**                                                                       |
+======================================+==============+=======================================================================================+
| :code:`micro_batch_size`             | int          | Micro batch size that fits on each GPU                                                |
+--------------------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`global_batch_size`            | int          | Global batch size considering gradient accumulation and data parallelism              |
+--------------------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`tensor_model_parallel_size`   | int          | Intra-layer model parallelism                                                         |
+--------------------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`pipeline_model_parallel_size` | int          | Inter-layer model parallelism                                                         |
+--------------------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`seed`                         | int          | Seed used in training                                                                 |
+--------------------------------------+--------------+---------------------------------------------------------------------------------------+

CLIP
~~~~~~~~

For model-specific configurations, refer to `clip <./clip.html#clip>`_.
