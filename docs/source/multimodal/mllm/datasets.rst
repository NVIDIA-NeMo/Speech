Multimodal Language Model Datasets
==================================

The NeMo Framework multimodal language model supports the conversation data format, drawing inspiration from and designed based on `LLaVA <https://github.com/haotian-liu/LLaVA/tree/main>`_. Sample datasets can be explored at `LLaVA's data documentation <https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md>`_.

Prepare the Training Dataset
----------------------------

The NeVA model training encompasses two phases: pretraining and fine-tuning. Each phase mandates a unique dataset.

For **pretraining**, utilize the *LAION/CC/SBU BLIP-Caption Concept-balanced 558K* dataset. Access this dataset via `LLaVA's GitHub <https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md>`_. After procuring the dataset, extract it to:

.. code-block:: bash

   /path/to/neva/datasets/LLaVA-Pretrain-LCS-558K/blip_laion_cc_sbu_558k.json

Acquire the image data from `Hugging Face <https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip>`__ and extract to:

.. code-block:: bash

   /path/to/neva/datasets/LLaVA-Pretrain-LCS-558K/images

For **fine-tuning**, deploy the *LLaVA-Instruct-150K* dataset. This is also available on `LLaVA's GitHub <https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md>`_. You can download the prompts from `HuggingFace <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main>`__:

.. code-block:: bash

   /path/to/neva/datasets/LLaVA-Instruct-150K/

Image data for this phase can be obtained from the `COCO Dataset <https://cocodataset.org/#download>`_. Once downloaded, extract the images to:

.. code-block:: bash

   /path/to/neva/datasets/LLaVA-Instruct-150K/images

Additional Preparation for the NeVA Model
-----------------------------------------

The following instructions are specific to the NeVA model within the NeMo Framework multimodal language models.

Set Up LLaMA-2 Chat Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Support is available for both the 7B and 13B chat models. Both can be downloaded from `LLaVA's Model Zoo <https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md>`__. After downloading the checkpoint you want from Hugging Face, extract and store it on your local system to prepare for pretraining.

To convert the LLaMA-2 checkpoints to NeMo's format, follow these steps:

1. Adjust the default YAML file at `megatron_llama_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/language_modeling/conf/megatron_llama_config.yaml>`__. Ensure ``model.mcore_gpt`` and ``model.transformer_engine`` are set to `False` before the checkpoint conversion.

2. For the 7B chat model, use this conversion command:

.. code-block:: bash

   python /opt/NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py \
     --in-file <PATH-TO-HF-CHECKPOINT> \
     --out-file /path/to/neva/checkpoints/llama-2-7b-chat.nemo

For the 13B model, adjust the paths in the `--in-file` and `--out-file` parameters accordingly.

3. Execute the subsequent command to divide the checkpoint for tensor model parallel sizes of 4 or 8. It's advisable to use TP=4 for the 7B model and TP=8 for the 13B model to ensure both pretraining and fine-tuning operate without memory complications.

.. code-block:: bash

   # Instructions for the 7B model partitioning provided here.
   # Adjust parameters for the 13B model as needed.
   python /opt/NeMo/examples/nlp/language_modeling/megatron_change_num_partitions.py \
     --model_file=/path/to/neva/checkpoints/llama-2-7b-chat.nemo  \
     --target_file=/path/to/neva/checkpoints/llama-2-7b-chat-tp4.nemo \
     --tensor_model_parallel_size=1 \
     --target_tensor_model_parallel_size=4 \
     --pipeline_model_parallel_size=1 \
     --target_pipeline_model_parallel_size=1 \
     --tp_conversion_only \
     --model_class="nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel" \
     --tokenizer_model_path=<PATH-TO-HF-CHECKPOINT>/tokenizer.model

Configure Tokenizer
^^^^^^^^^^^^^^^^^^^

For NeVA training, it is vital that you integrate special tokens into the tokenizer. After obtaining the 7B/13B model from Hugging Face, you need to procure the corresponding tokenizer model. Referring to the 7B-chat model:

1. Download the `tokenizer.model <https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview/blob/main/tokenizer.model>`_ to:

.. code-block:: bash

   /path/to/neva/tokenizers/tokenizer.model

2. Step 3 requires NeMo Framework to be installed. For quick setup, we recommend running it within the NeMo Framework container.

3. Employ the command below to infuse special tokens into the tokenizer:

.. code-block:: bash

   cd /opt; git clone https://github.com/google/sentencepiece.git && \
     cd sentencepiece && \
     mkdir build && \
     cd build && \
     cmake .. && \
     make && \
     make install && \
     ldconfig
   cd /opt/sentencepiece/src/; protoc --python_out=/opt/NeMo/scripts/tokenizers/ sentencepiece_model.proto
   python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py \
   --input_file /path/to/neva/tokenizers/tokenizer.model \
   --output_file /path/to/neva/tokenizers/tokenizer_neva.model \
   --is_userdefined \
   --tokens "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" \
            "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>"
