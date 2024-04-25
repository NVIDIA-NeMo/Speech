Converting from Megatron-LM
===========================

NVIDIA NeMo and NVIDIA Megatron-LM share many underlying technologies. This document provides guidance for migrating your project from Megatron-LM to NVIDIA NeMo.

Converting Checkpoints
----------------------

You can convert your GPT-style model checkpoints trained with Megatron-LM into the NeMo Framework using the provided example script. This script facilitates the conversion of Megatron-LM checkpoints to NeMo compatible formats.

.. code-block:: bash

   <NeMo_ROOT_FOLDER>/examples/nlp/language_modeling/megatron_lm_ckpt_to_nemo.py \
     --checkpoint_folder <path_to_PTL_checkpoints_folder> \
     --checkpoint_name megatron_gpt--val_loss=99.99-step={steps}-consumed_samples={consumed}.0 \
     --nemo_file_path <path_to_output_nemo_file> \
     --model_type <megatron_model_type> \
     --tensor_model_parallel_size <tensor_model_parallel_size> \
     --pipeline_model_parallel_size <pipeline_model_parallel_size> \
     --gpus_per_node <gpus_per_node>

Resuming Training
-----------------

To resume training from a converted Megatron-LM checkpoint, it is crucial to correctly set up the training parameters to match the previous learning rate schedule. Use the following setting for the `trainer.max_steps` parameter in your NeMo training configuration:

.. code-block:: none

   trainer.max_steps=round(lr-warmup-fraction * lr-decay-iters + lr-decay-iters)

This configuration ensures that the learning rate scheduler in NeMo continues from where it left off in Megatron-LM, using the `lr-warmup-fraction` and `lr-decay-iters` arguments from the original Megatron-LM training setup.

