Common Configuration Files
============================

This section describes the NeMo configuration file setup that is specific to models in the MM Text2Img collection. For general information
about how to set up and run experiments that is common to all NeMo models (e.g. Experiment Manager and PyTorch Lightning trainer
parameters), see the `Core Documentation <../../core/core.html>`_ section.

The model section of the NeMo Multimodal Text2Img configuration files generally requires information about the dataset(s) being used, 
the text and image encoder, parameters for any augmentation being performed, as well as the model architecture specification. The sections on
this page cover each of these in more detail.

Example configuration files for all of the NeMo Multimodal Text2Img scripts can be found in the
`config directory of the examples <PUT THE URL>`_.


Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``train``, ``validation``, and
``test`` sections in the configuration file, respectively. Depending on the task, there may be arguments specifying the augmentations
for the dataset, the resolution filter for filtering out images, and so on. 

Any initialization parameter that is accepted for the Dataset class used in the experiment can be set in the config file.
Refer to the `Datasets <../api.html#Datasets>`__ section of the API for a list of Datasets and their respective parameters.

An example Text2Img train configuration should look similar to the following:

.. code-block:: yaml

  model:
    data:
      num_workers: 16 # The number of workers for dataloader process
      train:
        dataset_path: # List of wdinfo files for the datasets to train on
          - dataset1.pkl
          - dataset2.pkl
        augmentations:
          resize_samllest_side: 64 # Resize the smallest side of the image to the specified resolution
          center_crop_h_w: 64, 64 # Center cropping
          horizontal_flip: False # Whether to perform horizontal flip
        filterings:
          resolution:
            method: larger
            value: 64
      webdataset:
        use_webdataset: True
        infinite_sampler: false
        local_root_path: ??? # Path that stores the dataset
        verbose: False # Whether to print detail debugging information

Currently, our diffusion-based Text2Img models do not require validation steps for faster convergence. 
As discussed in `Datasets <./datasets.html>`_, storing training dataset in webdataset format is the requirement for all
text2img training pipeline. Using ``webdataset.infinite_sampler=True`` is the preferred way for training especially if the dataset
is large as suggested by `Webdataset Multinode Training Guideline <https://github.com/webdataset/webdataset#multinode-training>`_ .
          
Enabling ``train.filterings`` allows one to filter out images (and corresponding text pairs) based on some common use cases (e.g., minimum resolution)
without having to create a redundant subset of the webdataset on the disk prior to training. The example above showcases how to filter the dataset so that only images with a resolution
larger than 64x64 will be used for training. Concatenating multiple webdataset is as easy as listing all wdinfo files in
``train.dataset_path``.




Trainer Configuration
--------------------------

Trainer configuration specifies the arguments for Pytorch Lightning Trainer Object.

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

Refer to the `Pytorch Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html#>`__ API section 
for all possible arguments


Experiment Manager Configurations
---------------------------

NeMo Experiment Manager provides convenient way to configure logging, saving, resuming options and more.

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
    create_checkpoint_callback: True  # Whether you want exp_manager to create a model checkpoint callback
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

EMA feature can be enabled by setting ``exp_manager.ema.enable=True``. 

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

By default we use ``fused_adam`` as the optimizer, refer to NeMo user guide for all supported optimizers.
Learning rate scheduler can be specified in ``optim.sched`` section.

Model Architecture Configurations
------------------------

Each configuration file should describe the model architecture being used for the experiment. 

Here is the list of the parameters in the model section which are shared among most of the MM Text2Img models:

+---------------------------+--------------+---------------------------------------------------------------------------------------+
| **Parameter**             | **Datatype** | **Description**                                                                       |
+===========================+==============+=======================================================================================+
| :code:`micro_batch_size`  | int          | micro batch size that fits on each GPU                                                |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`global_batch_size` | int          | global batch size that takes consideration of gradient accumulation, data parallelism |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`inductor`          | bool         | enable TorchInductor optimization                                                     |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`channels_last`     | bool         | enable NHWC training format                                                           |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`seed`              | int          | seed used in training                                                                 |
+---------------------------+--------------+---------------------------------------------------------------------------------------+
