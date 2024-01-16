Common Configuration Files
==========================

This section provides a detailed overview of the NeMo configuration file setup specific to models within the NeMo Multimodal Language Model collection. For foundational knowledge about setting up and executing experiments common to all NeMo models, such as the Experiment Manager and PyTorch Lightning trainer parameters, refer to the :doc:`../core/core` section.

Within the configuration files of the NeMo Multimodal Language Model, details concerning dataset(s), augmentation, optimization parameters, and model architectural specifications are central. This page explores each of these aspects.

Discover exemplary configuration files for all NeMo Multimodal Language Model scripts in the `config directory of the examples <https://TODOURL>`_.

Dataset Configuration
---------------------

The NeMo multimodal language model currently supports a conversation data format, inspired by and designed from https://github.com/haotian-liu/LLaVA/tree/main. To explore a sample dataset, visit https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md.

The configuration file allows setting any initialization parameter accepted by the Dataset class used in the experiment. For a comprehensive list of Datasets and their parameters, visit the `Datasets <./api.html#Datasets>`__ section of the API.

A typical training configuration is as follows:

.. code-block:: yaml

  data:
    num_workers: 8
    dataloader_type: cyclic
    data_path: path/to/conversations.json
    lazy_preprocess: True
    is_multimodal: True
    conv_template: llama_2
    image_token_len: 256
    image_folder: path/to/images
    image_aspect_ratio: 'square'

Key parameters include:

- ``data_path``: The path to the dataset in JSON format.
- ``is_multimodal``: Indicates if the dataset has multiple modalities (e.g., text and images).
- ``conv_template``: The template used for conversation format. Supports values like 'nvgpt' and 'llama_2'.
- ``image_token_len``: Specifies how many tokens in the language model word embedding each image will occupy.
- ``image_folder``: The path to the folder containing images related to the dataset.
- ``image_aspect_ratio``: Specifies whether to pad or crop the image to maintain the aspect ratio, such as 'square'.

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

+---------------------------+--------------+---------------------------------------------------------------------------------------+
| **Parameter**             | **Datatype** | **Description**                                                                       |
+===========================+==============+=======================================================================================+
| :code:`micro_batch_size`  | int          | micro batch size that fits on each GPU                                                |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`global_batch_size` | int          | global batch size that takes consideration of gradient accumulation, data parallelism |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`tensor_model_parallel_size`       | int         | intra-layer model parallelism                                                     |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`pipeline_model_parallel_size`     | int         | inter-layer model parallelism                                                           |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`seed`              | int          | seed used in training                                                                 |
+---------------------------+--------------+---------------------------------------------------------------------------------------+

NeVA
~~~~~~~~

For model-specific configurations, refer to `Neva <./neva.html#neva>`_.
