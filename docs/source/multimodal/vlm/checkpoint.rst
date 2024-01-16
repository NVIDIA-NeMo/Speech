Checkpoints
===========

In this section, we present four key functionalities of NVIDIA NeMo related to checkpoint management:

1. **Checkpoint Loading**: Use the :code:`restore_from()` method to load local ``.nemo`` checkpoint files.
2. **Partial Checkpoint Conversion**: Convert partially-trained ``.ckpt`` checkpoints to the ``.nemo`` format.
3. **Community Checkpoint Conversion**: Transition checkpoints from community sources, like HuggingFace, into the ``.nemo`` format.
4. **Model Parallelism Adjustment**: Modify model parallelism to efficiently train models that exceed the memory of a single GPU. NeMo employs both tensor (intra-layer) and pipeline (inter-layer) model parallelisms. Dive deeper with `"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" <https://arxiv.org/pdf/2104.04473.pdf>`_. This tool aids in adjusting model parallelism, accommodating users who need to deploy on larger GPU arrays due to memory constraints.

Understanding Checkpoint Formats
--------------------------------

A ``.nemo`` checkpoint is fundamentally a tar file that bundles the model configurations (given as a YAML file), model weights, and other pertinent artifacts like tokenizer models or vocabulary files. This consolidated design streamlines sharing, loading, tuning, evaluating, and inference.

Contrarily, the ``.ckpt`` file, created during PyTorch Lightning training, encompasses both the model weights and the optimizer states, usually employed to pick up training from a pause.

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

Only the last checkpoint is automatically saved in the ``.nemo`` format. If intermediate training checkpoints evaluation is required, a ``.nemo`` conversion might be necessary. For this, refer to the script at `script <http://TODOURL>`_:

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

CLIP Checkpoints
^^^^^^^^^^^^^^^^

To migrate community checkpoints:

.. code-block:: python

   python examples/multimodal/foundation/clip/convert_external_clip_to_nemo.py \
       --arch=ViT-H-14 \
       --version=laion2b_s32b_b79k \
       --hparams_file=path/to/saved.yaml \
       --nemo_file_path=open_clip.nemo

Ensure the NeMo hparams file has the correct model architectural parameters, placed at `path/to/saved.yaml`. An example can be found in `examples/multimodal/foundation/clip/conf/megatron_clip_config.yaml`.

For OpenCLIP migrations, provide the architecture (`arch`) and version (`version`) according to the OpenCLIP `model list <https://github.com/mlfoundations/open_clip#usage>`_. For Hugging Face conversions, set the version to `huggingface` and the architecture (`arch`) to the specific Hugging Face model identifier, e.g., `yuvalkirstain/PickScore_v1`.

Model Parallelism Adjustment
---------------------------

CLIP Checkpoints
^^^^^^^^^^^^^^^^

To adjust model parallelism from original model parallelism size to a new model parallelism size (Note: NeMo CLIP currently only supports `pipeline_model_parallel_size=1`):

.. code-block:: python

   python examples/nlp/language_modeling/megatron_change_num_partitions.py \
    --model_file=/path/to/source.nemo \
    --target_file=/path/to/target.nemo \
    --tensor_model_parallel_size=??? \
    --target_tensor_model_parallel_size=??? \
    --pipeline_model_parallel_size=-1 \
    --target_pipeline_model_parallel_size=1 \
    --precision=32 \
    --model_class="nemo.collections.multimodal.models.clip.megatron_clip_models.MegatronCLIPModel" \
    --tp_conversion_only
