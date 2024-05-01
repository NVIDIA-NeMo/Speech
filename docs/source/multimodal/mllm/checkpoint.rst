Checkpoints
===========

In this section, we present four key functionalities of NVIDIA NeMo related to checkpoint management:

1. **Checkpoint Loading**: Load local ``.nemo`` checkpoint files with the :code:`restore_from()` method.
2. **Partial Checkpoint Conversion**: Convert partially-trained ``.ckpt`` checkpoints to the ``.nemo`` format.
3. **Community Checkpoint Conversion**: Transition checkpoints from community sources, like HuggingFace, into the ``.nemo`` format.
4. **Model Parallelism Adjustment**: Modify model parallelism to efficiently train models that exceed the memory of a single GPU. NeMo employs both tensor (intra-layer) and pipeline (inter-layer) model parallelisms. Dive deeper with "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (`link <https://arxiv.org/pdf/2104.04473.pdf>`_). This tool aids in adjusting model parallelism, accommodating users who need to deploy on larger GPU arrays due to memory constraints.

Understanding Checkpoint Formats
--------------------------------

A ``.nemo`` checkpoint is fundamentally a tar file that bundles the model configurations (given as a YAML file), model weights, and other pertinent artifacts like tokenizer models or vocabulary files. This consolidated design streamlines sharing, loading, tuning, evaluating, and inference.

On the other hand, the ``.ckpt`` file is a product of PyTorch Lightning training. It stores model weights and optimizer states, and it's generally used for resuming training.

Subsequent sections delve into each of the previously listed functionalities, emphasizing the loading of fully trained checkpoints for evaluation or additional fine-tuning.


Loading Local Checkpoints
-------------------------

NeMo inherently saves any model's checkpoints in the ``.nemo`` format. To manually save a model at any stage:

.. code-block:: python

   model.save_to(<checkpoint_path>.nemo)

To load a local ``.nemo`` checkpoint:

.. code-block:: python

   import nemo.collections.multimodal as nemo_multimodal
   model = nemo_multimodal.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

Replace `<MODEL_BASE_CLASS>` with the appropriate MM model class.

Converting Local Checkpoints
----------------------------

The training script only auto-converts the final checkpoint into the ``.nemo`` format. To evaluate intermediate training checkpoints, conversion to ``.nemo`` might be needed. For this:

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
       examples/multimodal/convert_ckpt_to_nemo.py \
       --checkpoint_folder <path_to_PTL_checkpoints_folder> \
       --checkpoint_name <checkpoint_name> \
       --nemo_file_path <path_to_output_nemo_file> \
       --tensor_model_parallel_size <tensor_model_parallel_size> \
       --pipeline_model_parallel_size <pipeline_model_parallel_size>

Converting Community Checkpoints
--------------------------------

NeVA Checkpoints
^^^^^^^^^^^^^^^^

Currently, the conversion mainly supports LLaVA checkpoints based on "llama-2 chat" checkpoints. As a reference, we'll consider the checkpoint `llava-llama-2-13b-chat-lightning-preview <https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview>`_.

After downloading this checkpoint and saving it at ``/path/to/llava-llama-2-13b-chat-lightning-preview``, undertake the following procedures:

Modifying the Tokenizer
"""""""""""""""""""""""

NeMo mandates adding specific tokens to the tokenizer model for peak performance. To modify an existing tokenizer located in ``/path/to/llava-llama-2-13b-chat-lightning-preview/tokenizer``, execute the following in the NeMo container:

.. code-block:: bash

   cd /opt/sentencepiece/src/
   protoc --python_out=/opt/NeMo/scripts/tokenizers/ sentencepiece_model.proto
   python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py \
   --input_file /path/to/llava-llama-2-13b-chat-lightning-preview/tokenizer.model \
   --output_file /path/to/llava-llama-2-13b-chat-lightning-preview/tokenizer_neva.model \
   --is_userdefined \
   --tokens "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" \
            "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>"

Checkpoint Conversion
"""""""""""""""""""""

For conversion:

.. code-block:: bash

   python examples/multimodal/mllm/neva/convert_hf_llava_to_neva.py \
     --in-file /path/to/llava-llama-2-13b-chat-lightning-preview \
     --out-file /path/to/neva-llava-llama-2-13b-chat-lightning-preview.nemo \
     --tokenizer-model /path/to/llava-llama-2-13b-chat-lightning-preview/tokenizer_add_special.model
     --conv-template llama_2


Model Parallelism Adjustment
----------------------------

NeVA Checkpoints
^^^^^^^^^^^^^^^^

Adjust model parallelism with:

.. code-block:: bash

   python examples/nlp/language_modeling/megatron_change_num_partitions.py \
    --model_file=/path/to/source.nemo \
    --target_file=/path/to/target.nemo \
    --tensor_model_parallel_size=??? \
    --target_tensor_model_parallel_size=??? \
    --pipeline_model_parallel_size=??? \
    --target_pipeline_model_parallel_size=??? \
    --model_class="nemo.collections.multimodal.models.multimodal_llm.neva.neva_model.MegatronNevaModel" \
    --precision=32 \
    --tokenizer_model_path=/path/to/tokenizer.model \
    --tp_conversion_only
