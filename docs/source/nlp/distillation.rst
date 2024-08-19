.. _megatron_distillation:

Distillation
==========================

Knowledge Distillation (KD)
--------------------------------

KD involves using information from an existing trained model to train a second (usually smaller, faster) model, thereby "distilling" knowledge from one to the other.

Distillation has two primary benefits: faster convergence and higher end accuracy than traditional training.

In NeMo, distillation is enabled by the `NVIDIA TensorRT Model Optimizer (ModelOpt) <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ library -- a library to optimize deep-learning models for inference on GPUs.

The logits-distillation process consists of the following steps:

1. Loading both student and teacher model checkpoints (must support same parallelism strategy, if any)
2. Training until convergence, where forward passes are run on both models (and backward only on student), performing a specific loss function between the logits.
3. Saving the final student model.


Example
^^^^^^^
The example below shows how to run the distillation script for LLama models.

The script must be launched correctly with the number of processes equal to tensor parallelism. This is achieved with the ``torchrun`` command below:

.. code-block:: bash

    STUDENT_CKPT="path/to/student.nemo"  # can also be None (will use default architecture found in examples/nlp/language_modeling/conf/megatron_llama_distill.yaml)
    TEACHER_CKPT="path/to/teacher.nemo"
    TOKENIZER="path/to/tokenizer.model"
    DATA_PATHS="[1.0,path/to/tokenized/data]"
    FINAL_SAVE_FILE="final_checkpoint.nemo"
    TP=4

    NPROC=$TP
    launch_config="torchrun --nproc_per_node=$NPROC"

    ${launch_config} examples/nlp/language_modeling/megatron_gpt_distillation.py \
        model.restore_from_path=$STUDENT_CKPT \
        model.kd_teacher_restore_from_path=$TEACHER_CKPT \
        model.tensor_model_parallel_size=$TP \
        model.tokenizer.model=$TOKENIZER \
        model.data.data_prefix=$DATA_PATHS \
        model.nemo_path=$FINAL_SAVE_FILE \
        trainer.precision=bf16 \
        trainer.devices=$NPROC

For large models, the command can be used in multi-node setting. For example, this can be done with `NeMo Framework Launcher <https://github.com/NVIDIA/NeMo-Framework-Launcher>`_ using Slurm.


Limitations
^^^^^^^^^^^
* Only Megatron Core-based GPT models are supported
* Only logit-pair distillation is supported for now
* Pipeline parallelism not yet supported
* FSDP strategy not yet supported
