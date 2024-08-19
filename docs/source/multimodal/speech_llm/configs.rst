Common Configuration Files
==========================

This section provides a detailed overview of the NeMo Framework configuration file setup, specifically for models within the NeMo Speech-augmented Large Language Models (SpeechLLM) collection. For foundational knowledge on setting up and executing experiments common to all NeMo Framework models, including the Experiment Manager and PyTorch Lightning trainer parameters, refer to the :doc:`core <../../core/core>` documentation.

The configuration files for NeMo SpeechLLMs focus on key details such as datasets, augmentation, optimization parameters, and model architectural specifications. This page explores each of these aspects.

Discover exemplary configuration files for all SpeechLLMs in the `config directory of the examples <https://github.com/NVIDIA/NeMo/tree/main/examples/multimodal/speech_llm/conf>`_.


Dataset Configuration
---------------------

The dataset configuration is based on the NeMo ASR data configuration and the NLP data configuration.

The configuration file enables you to set any initialization parameter accepted by the Dataset class used in the experiment. For a comprehensive list of datasets and their parameters, refer to the Datasets section of the :doc:`API <./api>`.

A typical training configuration is as follows:

.. code-block:: yaml

    train_ds:
        manifest_filepath: ??? # Path to a list of JSONL files corresponding to the source data.
        global_batch_size: 4
        micro_batch_size: 2
        shuffle: True
        num_workers: 0
        pin_memory: True
        max_seq_length: 2048
        min_seq_length: 1
        drop_last: True
        concat_sampling_probabilities: null # When providing a list of datasets, this arg defines the sampling probabilities from each dataset when strategy='random'
        context_key: 'context'
        answer_key: 'answer'
        add_eos: True
        add_eos: False
        end_string: null
        add_sep: False
        add_bos: False
        separate_prompt_and_response_with_newline: False
        truncation_field: "context" # Options: ['context', 'answer']
        prompt_template: "Q: {context}\nA: {answer}" # fstring to use for assistant prompt. Example: "Q: {input}\nA: {output}"
        # ASR configs
        sample_rate: 16000 #${model.audio_encoder.preprocessor.sample_rate}
        max_duration: 24 # it is set for LibriSpeech, you may need to update it for your dataset
        min_duration: 0.1
        # tarred datasets
        is_tarred: false
        tarred_audio_filepaths: null
        shuffle_n: 2048
        # bucketing params
        bucketing_strategy: "fully_randomized"
        bucketing_batch_size: null
        # multi-audio configs
        audio_locator: null


The key configuration parameters include:

- ``manifest_filepath``: The path to the dataset in JSON lines format, where each line in the file is a Python dictionary. This can either be a single file or a list of files.
- ``global_batch_size``: The global batch size that takes consideration of gradient accumulation, data parallelism.
- ``micro_batch_size``: The micro batch size that fits on each GPU.
- ``shuffle``: Whether to shuffle the dataset.
- ``num_workers``: The number of workers to use for data loading.
- ``pin_memory``: Whether to pin memory for faster data transfer.
- ``max_seq_length``: The maximum sequence length for LLM.
- ``min_seq_length``: The minimum sequence length for LLM.
- ``drop_last``: Whether to drop the last batch if it is smaller than the batch size.
- ``context_key``: The key in the JSON line that corresponds to the context used for LLM input.
- ``answer_key``: The key in the JSON line that corresponds to the answer used for groundtruth.
- ``add_eos``: Whether to add an end-of-sequence token.
- ``add_bos``: Whether to add a beginning-of-sequence token.
- ``add_sep``: Whether to add a separator token.
- ``end_string``: The string to used to trigger end of generation, default to null to use EOS token.
- ``separate_prompt_and_response_with_newline``: Whether to separate the prompt and response with a newline.
- ``truncation_field``: The field to truncate if the sequence length exceeds the maximum sequence length.
- ``prompt_template``: The fstring to use for the LLM prompt, where the context and answer will be formatted.
- ``sample_rate``: The sample rate of the audio data.
- ``max_duration``: The maximum duration of the audio data to be included.
- ``min_duration``: The minimum duration of the audio data to be included.
- ``is_tarred``: Whether the dataset is tarred.
- ``tarred_audio_filepaths``: The path to the tarred audio files.
- ``shuffle_n``: The number of samples to shuffle in tarred datasets, not used for non-tarred datasets.
- ``bucketing_strategy``: The strategy to use for bucketing, options include 'fully_randomized', 'synced_randomized'.
- ``bucketing_batch_size``: The batch size to use for each bucket, if not provided, the micro batch size is used.
- ``audio_locator``: The special string to locate the position of each audio to be put in the text prompt.


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

The NeMo Framework Experiment Manager provides a streamlined approach to manage various tasks such as logging, saving, and resuming.

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

NeMo Framework offers a variety of optimizers to enhance the training of neural network models. The following example shows the ``fused_adam`` default optimizer. The learning rate scheduler can be specified in the ``optim.sched`` section.

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

For more information on the supported optimizers, refer to the "Optimization" section in the NeMo APIs :doc:`docs <../../core/core>`.

Model Configurations
--------------------

Each configuration file should detail the model architecture used for the experiment.

The following table shows the parameters commonly shared across most multimodal language models.

+------------------------------------------+--------------+---------------------------------------------------------------------------------------+
| **Parameter**                            | **Datatype** | **Description**                                                                       |
+===========================+==============+==============+=======================================================================================+
| :code:`micro_batch_size`                 | int          | micro batch size that fits on each GPU                                                |
+------------------------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`global_batch_size`                | int          | global batch size that takes consideration of gradient accumulation, data parallelism |
+------------------------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`tensor_model_parallel_size`       | int          | intra-layer model parallelism                                                         |
+------------------------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`pipeline_model_parallel_size`     | int          | inter-layer model parallelism                                                         |
+------------------------------------------+--------------+---------------------------------------------------------------------------------------+
| :code:`seed`                             | int          | seed used in training                                                                 |
+------------------------------------------+--------------+---------------------------------------------------------------------------------------+

Speech-Augmented Language Model (SALM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For information about SALM model-specific configurations, refer to `the examples <https://github.com/NVIDIA/NeMo/tree/main/examples/multimodal/speech_llm/conf/salm>`__.


BESt features from TwO Worlds (BESTOW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For information about BESTOW model-specific configurations, refer to `the examples <https://github.com/NVIDIA/NeMo/tree/main/examples/multimodal/speech_llm/conf/bestow>`__.
