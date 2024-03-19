Community Model Converter User Guide
====================================

This guide provides instructions on how to use the conversion scripts to convert models between Community model and NVIDIA's NeMo format.

Support Matrix
--------------

+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Conversion           | From             | To              | Github Link                                                                                                        |
+======================+==================+=================+====================================================================================================================+
| Baichuan             | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_baichuan2_hf_to_nemo.py>`_   |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Baichuan             | NeMo             | Hugging Face    | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_baichuan2_nemo_to_hf.py>`_   |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| BERT                 | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_bert_hf_to_nemo.py>`_        |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| BERT                 | NeMo             | Hugging Face    | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_bert_nemo_to_hf.py>`_        |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Falcon               | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_falcon_hf_to_nemo.py>`_      |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Falcon               | NeMo             | Hugging Face    | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_falcon_nemo_to_hf.py>`_      |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Gemma                | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_gemma_hf_to_nemo.py>`_       |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Gemma                | JAX              | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_gemma_jax_to_nemo.py>`_      |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Gemma                | PyTorch          | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_gemma_pyt_to_nemo.py>`_      |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| GPT                  | NeMo             | mcore           | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_gpt_nemo_to_mcore.py>`_      |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| LLaMA                | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py>`_       |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| LLaMA                | NeMo             | Hugging Face    | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py>`_       |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Mistral 7B           | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_mistral_7b_hf_to_nemo.py>`_  |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Mistral 7B           | NeMo             | Hugging Face    | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_mistral_7b_nemo_to_hf.py>`_  |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Mixtral              | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_mixtral_hf_to_nemo.py>`_     |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Mixtral              | NeMo             | Hugging Face    | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_mixtral_nemo_to_hf.py>`_     |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| MPT                  | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_mpt_hf_to_nemo.py>`_         |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Starcoder            | Hugging Face     | NeMo            | `Link <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_starcoder_hf_to_nemo.py>`_   |
+----------------------+------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+


Convert Hugging Face LLaMA Checkpoints to NeMo
----------------------------------------------

To convert a Hugging Face LLaMA checkpoint into a NeMo checkpoint, use the following command:

.. code-block:: bash

    python convert_llama_hf_to_nemo.py>`_ \
     --input_name_or_path <path_to_hf_checkpoints_folder> \
     --output_path <path_to_output_nemo_file>

Convert NeMo Checkpoint to Hugging Face LLaMA
---------------------------------------------

To convert a NeMo checkpoint into a Hugging Face LLaMA checkpoint, you have two options:

1. Generate only the Hugging Face weights:

.. code-block:: bash

    python convert_<model>_nemo_to_hf.py>`_ \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin

2. Generate the full Hugging Face model folder:

.. code-block:: bash

    python convert_<model>_nemo_to_hf.py>`_ \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/model_folder \
    --hf_input_path /path/to/input_hf_folder \
    --hf_output_path /path/to/output_hf_folder

Replace `<model>` with the specific model you are converting.

Use the ``--cpu-only`` flag if the model cannot fit in the GPU, such as for Llama2 70b models. Note that using this option will significantly slow down the conversion process.

Command-Line Arguments
----------------------

- ``--input_name_or_path``: Path to the input .nemo file or the Hugging Face model folder.
- ``--output_path``: Path to the output file or folder, depending on the conversion direction.
- ``--hf_input_path``: (Optional) Path to the input Hugging Face model folder.
- ``--hf_output_path``: (Optional) Path to the output Hugging Face model folder.
