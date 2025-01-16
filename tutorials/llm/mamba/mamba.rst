Mamba2 and Mamba2-Transformer Hybrid Models Fine-Tuning
=======================================================

`State Space Models (SSMs) <https://arxiv.org/pdf/2405.21060>`__ have recently emerged as a promising alternative to transformers. SSMs offer advantages such as linear time complexity relative to sequence length and a constant cache size for inference. These features enable the processing of longer sequences and higher throughput. Despite these benefits, SSMs alone may fall short compared to transformers on tasks that demand strong copying or in-context learning capabilities.

To harness the strengths of both approaches, SSM-Hybrid models incorporate MLP, Transformer, and SSM blocks in their architecture. As highlighted in `a study by NVIDIA <https://arxiv.org/pdf/2406.07887>`__, these hybrid models outperform traditional transformers of the same size by achieving faster inference times due to the inclusion of SSM blocks. Based on experimental results, Mamba2-Hybrid models not only surpass transformer baselines in performance but also benefit from increased computational efficiency.

The Mamba2 models discussed in the `Transformers are SSMs <https://arxiv.org/pdf/2405.21060>`__ paper are available in five different sizes: 130 million, 370 million, 780 million, 1.3 billion, and 2.7 billion parameters. The Mamba2-Hybrid models, along with their Mamba2 baseline as released by `NVIDIA <https://arxiv.org/pdf/2406.07887>`__, are provided in an 8 billion parameter size.

`NVIDIA NeMo
Framework <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>`__ provides tools to perform Fine-tuning on Mamba2 and Mamba2-Hybrid to fit your use case.

Requirements
-------------

In order to proceed, ensure that you have met the following requirements:

* Full Fine-Tuning System Configuration
    * Small models (130m, 370m, 780m)
        * Access to at least 1 NVIDIA GPU with a cumulative memory of at least 40GB, for example: 1 x A6000-40GB.

    * Mid-size models (1.3b, 2.7b)
        * Access to at least 1 NVIDIA GPU with a cumulative memory of at least 80GB, for example: 1 x H100-80GB or 1 x A100-80GB.

    * Large models (8b)
        * Access to at least 2 NVIDIA GPUs with a cumulative memory of at least 80GB, for example: 2 x H100-80GB or 2 x A100-80GB.


* A Docker-enabled environment, with `NVIDIA Container Runtime <https://developer.nvidia.com/container-runtime>`_ installed, which will make the container GPU-aware.


* `Authenticate with NVIDIA NGC <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-authentication>`_, generate API KEY from `NGC <https://org.ngc.nvidia.com/setup >`__, add the key to your credentials following instructions in `this guide <https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html>`__, and get into NVIDIA NeMo dev container ``nvcr.io/nvidia/nemo:dev``.

Step-by-step Guide for Fine-Tuning 
----------------------------------

Checkpoints from HuggingFace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Obtain the desired checkpoint from HuggigFace. The checkpoints below have different arrangement and there are a few preprocessing step for each.

1. `Repository <https://huggingface.co/collections/nvidia/ssms-666a362c5c3bb7e4a6bcfb9c>`__  for the Mamba2 and Mamba2-Hybrid models by `NVIDIA <https://arxiv.org/pdf/2406.07887>`__.
   The checkpoint from this repository is located in files tab under ``release/mp_rank_00/model_optim_rng.pt``. The tokenizer is under files tab and is named ``mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model``. You need both of these for conversion to ``.nemo`` checkpoint.

2. `Repository <https://huggingface.co/state-spaces>`__  for the Mamba2 models from the `Transformers are SSMs paper <https://arxiv.org/pdf/2405.21060>`__.
    For checkpoints from this repository, run the following python script to convert the pytorch checkpoint (`pytorch_model.bin` in the HuggingFace model card) to a format similar to the 8b models:

    .. code:: python
        
        import torch
        import os

        ckpt_path = "/path/to/pytorch_model.bin"
        pyt_checkpoint = torch.load(ckpt_path)
        new_ckpt_path = os.path.join(os.path.dirname(ckpt_path), f"wrapped_{os.path.basename(ckpt_path)}")
        
        # Save the new checkpoint which will be used as the input to the conversion script
        torch.save({"model": pyt_checkpoint}, new_ckpt_path)

    You will use this ``wrapped_pytorch_model.bin`` for the conversion to ``.nemo`` in the next step.



Convert the Pytorch Checkpoint to a NeMo Checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Get into the NVIDIA dev container from `NGC <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags>`_, or the 24.07 container (once released).

2. Run the conversion script from <SCRIPT-PATH>. For this conversion script, you should provide the checkpoint (and tokenizer in the case of 8b models) from the previous step for ``input_name_or_path``.

.. code:: bash

    CUDA_VISIBLE_DEVICES="0" python /opt/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py \
                                    --input_name_or_path <path to the source pytorch model> \
                                    --output_path <path to target .nemo model> \
                                    --mamba_ssm_ngroups 8 \
                                    --precision bf16 \
                                    --tokenizer_model_dir=<path to tokenizer.model> # Remove this line (or set it to None) for 130m, 370m, 780m, 1.3b, and 2.7b models.
                                    

* Note: the ``mamba_ssm_ngroups`` parameter should be 1 for the Mamba2 models from the `Transformers are SSMs paper <https://arxiv.org/pdf/2405.21060>`__ (130m, 370m, 780m, 1.3b, and 2.7b) and 8 for the Mamba2 and Mamba2-Hybrid models by `NVIDIA <https://arxiv.org/pdf/2406.07887>`__ (both 8b).

Run Fine-Tuning
^^^^^^^^^^^^^^^
1. Follow the steps from `here <https://nemo-framework-tme.gitlab-master-pages.nvidia.com/documentation/user-guide/latest/llms/gemma/dataprep.html>`__ to obtain and preprocess the fine-tuning dataset.

2. For full fine-tuning, run the following script

.. code:: bash

    #!/bin/bash

    MBS=4
    GBS=128
    TP=4 # According to the saved checkpoint
    SP=True # True only if TP>1 otherwise False
    SEQ_LEN=2048
    NUM_DEVICES=8
    PATH_TO_NEMO_MODEL=<path to .nemo file>
    TRAIN_DATASET_PATH=<path to training dataset file>
    VAL_DATASET_PATH=<path to validation dataset file>
    CONFIG_PATH="/opt/NeMo/examples/nlp/language_modeling/tuning/conf/"
    CONFIG_NAME="megatron_mamba_finetuning_config"
    SAVE_DIR=<path to the saving directory>

    torchrun --nproc_per_node=${NUM_DEVICES} \
            /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_mamba_finetuning.py \
            --config-path=${CONFIG_PATH} \
            --config-name=${CONFIG_NAME} \
            trainer.devices=${NUM_DEVICES} \
            trainer.precision=bf16 \
            trainer.accelerator=gpu \
            trainer.log_every_n_steps=1 \
            trainer.val_check_interval=100 \
            trainer.limit_val_batches=50 \
            +trainer.num_sanity_val_steps=0 \
            +trainer.accumulate_grad_batches=1 \
            trainer.max_steps=700 \
            trainer.gradient_clip_val=1.0 \
            exp_manager.exp_dir=${SAVE_DIR} \
            exp_manager.resume_if_exists=True \
            exp_manager.create_checkpoint_callback=True \
            exp_manager.create_wandb_logger=True \
            model.tensor_model_parallel_size=${TP} \
            model.sequence_parallel=$SP \
            model.peft.peft_scheme='none' \
            model.megatron_amp_O2=True \
            model.encoder_seq_length=${SEQ_LEN} \
            model.attention_backend='fused' \
            model.data.validation_ds.pad_to_max_length=True \
            model.data.train_ds.pad_to_max_length=True \
            model.optim.name="distributed_fused_adam" \
            model.data.train_ds.max_seq_length=${SEQ_LEN} \
            model.data.validation_ds.max_seq_length=${SEQ_LEN} \
            model.micro_batch_size=${MBS} \
            model.global_batch_size=${GBS} \
            model.restore_from_path=${PATH_TO_NEMO_MODEL} \
            model.data.train_ds.file_names=[${TRAIN_DATASET_PATH}] \
            model.data.validation_ds.file_names=[${VAL_DATASET_PATH}] \
            model.optim.lr=5e-6 \
            model.optim.sched.min_lr=1e-7


Evaluating the Fine-Tuned Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    #!/bin/bash

    MBS=32
    GBS=64
    TP=4 # According to the fine-tuned checkpoint
    SP=True # True only if TP>1 otherwise False
    SEQ_LEN=2048
    NUM_DEVICES=8
    PATH_TO_NEMO_MODEL=<path to .nemo file>
    TEST_DATASET="[<path to test datasets (list)>]"
    CONFIG_PATH="/opt/NeMo/examples/nlp/language_modeling/tuning/conf/"
    CONFIG_NAME="megatron_mamba_finetuning_config"
    SAVE_DIR=<path to the saving directory>

    CONFIG_PATH="/opt/NeMo/examples/nlp/language_modeling/tuning/conf/"
    CONFIG_NAME="megatron_mamba_generate_config"

    torchrun --nproc_per_node=${NUM_DEVICES}  /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_mamba_generate.py \
            --config-path=${CONFIG_PATH} \
            --config-name=${CONFIG_NAME} \
            trainer.devices=${NUM_DEVICES} \
            trainer.precision=bf16 \
            trainer.accelerator=gpu \
            trainer.log_every_n_steps=1 \
            trainer.val_check_interval=10 \
            trainer.limit_val_batches=20 \
            ++trainer.num_sanity_val_steps=0 \
            ++trainer.accumulate_grad_batches=1 \
            trainer.max_steps=1000 \
            trainer.gradient_clip_val=1.0 \
            exp_manager.exp_dir=${SAVE_DIR} \
            exp_manager.resume_if_exists=False \
            exp_manager.create_wandb_logger=False \
            model.attention_backend='fused' \
            model.megatron_amp_O2=True \
            model.peft.restore_from_path=False \
            +model.peft.restore_from_ckpt.checkpoint_dir=False \
            +model.peft.restore_from_ckpt.checkpoint_name=False \
            model.tensor_model_parallel_size=${TP} \
            model.micro_batch_size=${MBS} \
            model.global_batch_size=${GBS} \
            model.restore_from_path=${PATH_TO_NEMO_MODEL} \
            model.data.test_ds.file_names=${TEST_DATASET} \
            model.data.test_ds.names=["squad"] \
            model.data.test_ds.global_batch_size=${GBS} \
            model.data.test_ds.micro_batch_size=${MBS} \
            model.data.test_ds.tokens_to_generate=30 \
            model.answer_only_loss=True \
            inference.greedy=True \
            exp_manager.checkpoint_callback_params.monitor=validation_loss \
            ++inference.verbose=True \
            model.data.test_ds.write_predictions_to_file=True \
            model.data.test_ds.output_file_path_prefix=${SAVE_DIR}/shorteval \
            && echo "Eval finished, calculating scores" \
            && python /opt/NeMo/scripts/metric_calculation/peft_metric_calc.py --label_field original_answers \
            --pred_file ${SAVE_DIR}/shorteval_test_squad_inputs_preds_labels.jsonl > ${SAVE_DIR}/shorteval_test_squad_inputs_preds_labels.score \
            && cat ${SAVE_DIR}/shorteval_test_squad_inputs_preds_labels.score


Inference
^^^^^^^^^

For running inference on a Mamba model, one should use ``megatron_mamba_eval.py`` script. This evaluation script currently requires tensor/model parallel (TP1) of size one. If your checkpoint has TP>1, use the TP conversion step from above and set ``target_tensor_model_parallel_size=1``. The following is an example for using evaluation script:

.. code:: bash

    #!/bin/bash

    CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 /opt/NeMo/examples/nlp/language_modeling/megatron_mamba_eval.py \
            mamba_model_file=<path to .nemo checkpoint> \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=1 \
            inference.min_tokens_to_generate=64 \
            inference.tokens_to_generate=128 \
            prompts=["Why must not we look directly at the sun during a solar eclipse?"]
