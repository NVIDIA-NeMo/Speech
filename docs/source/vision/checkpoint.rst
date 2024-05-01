Checkpoints
===========

In this section, we present four key functionalities of NVIDIA NeMo related to checkpoint management:

1. **Checkpoint Loading**: Use the :code:`restore_from()` method to load local ``.nemo`` checkpoint files.
2. **Partial Checkpoint Conversion**: Convert partially-trained ``.ckpt`` checkpoints to the ``.nemo`` format.
3. **Community Checkpoint Conversion**: Convert checkpoints from community sources, like HuggingFace, into the ``.nemo`` format.
4. **Model Parallelism Adjustment**: Adjusting model parallelism is crucial when training models that surpass the memory capacity of a single GPU, such as the NVGPT 5B version, LLaMA2 7B version, or larger models. NeMo incorporates both tensor (intra-layer) and pipeline (inter-layer) model parallelisms. For a deeper understanding, refer to "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (`link <https://arxiv.org/pdf/2104.04473.pdf>`_). This tool assists in modifying model parallelism. After downloading and converting a community checkpoint to the ``.nemo`` format, if a user wishes to fine-tune it further, this adjustment might become essential.

Understanding Checkpoint Formats
--------------------------------

A ``.nemo`` checkpoint is fundamentally a tar file that bundles the model configurations (given as a YAML file), model weights, and other pertinent artifacts like tokenizer models or vocabulary files. This consolidated design streamlines sharing, loading, tuning, evaluating, and inference.

On the other hand, the ``.ckpt`` file, created during PyTorch Lightning training, contains only the model weights and the optimizer states, which is used to pick up training from a pause.

The subsequent sections elucidate instructions for the functionalities above, specifically tailored for deploying fully trained checkpoints for assessment or additional fine-tuning.

Loading Local Checkpoints
-------------------------

By default, NeMo saves checkpoints of trained models in the ``.nemo`` format. To save a model manually during training, use:

.. code-block:: python

   model.save_to(<checkpoint_path>.nemo)

To load a local ``.nemo`` checkpoint:

.. code-block:: python

   import nemo.collections.multimodal as nemo_multimodal
   model = nemo_multimodal.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

Replace `<MODEL_BASE_CLASS>` with the appropriate MM model class.

Converting Local Checkpoints
----------------------------

Only the last checkpoint is automatically saved in the ``.nemo`` format. If intermediate training checkpoints evaluation is required, a ``.nemo`` conversion might be necessary. For this, refer to the script at `<ADD convert_ckpt_to_nemo.py PATH>`:

.. code-block:: python

   python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
       examples/multimodal/convert_ckpt_to_nemo.py \
       --checkpoint_folder <path_to_PTL_checkpoints_folder> \
       --checkpoint_name <checkpoint_name> \
       --nemo_file_path <path_to_output_nemo_file> \
       --tensor_model_parallel_size <tensor_model_parallel_size> \
       --pipeline_model_parallel_size <pipeline_model_parallel_size>

Converting Community Checkpoints
--------------------------------

There is no support for converting community checkpoints to NeMo ViT.

Model Parallelism Adjustment
----------------------------

ViT Checkpoints
^^^^^^^^^^^^^^^^

To adjust model parallelism from original model parallelism size to a new model parallelism size (Note: NeMo ViT currently only supports `pipeline_model_parallel_size=1`):

.. code-block:: bash

   python examples/nlp/language_modeling/megatron_change_num_partitions.py \
    --model_file=/path/to/source.nemo \
    --target_file=/path/to/target.nemo \
    --tensor_model_parallel_size=??? \
    --target_tensor_model_parallel_size=??? \
    --pipeline_model_parallel_size=-1 \
    --target_pipeline_model_parallel_size=1 \
    --precision=32 \
    --model_class="nemo.collections.vision.models.megatron_vit_classification_models.MegatronVitClassificationModel" \
    --tp_conversion_only
