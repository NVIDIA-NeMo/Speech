Checkpoints
===========


In this section, we present key functionalities of NVIDIA NeMo related to checkpoint management.

Checkpoint Formats
------------------

A ``.nemo`` checkpoint is fundamentally a tar file that bundles the model configurations (specified inside a YAML file), model weights (inside a ``.ckpt`` file), and other artifacts like tokenizer models or vocabulary files. This consolidated design streamlines sharing, loading, tuning, evaluating, and inference.

In contrast, the ``.ckpt`` file, created during PyTorch Lightning training, contains both the model weights and the optimizer states, and is usually used to resume training.

Sharded Model Weights
---------------------

Within ``.nemo`` or ``.ckpt`` checkpoints, the model weights could be saved in either a regular format (one file called ``model_weights.ckpt`` inside model parallelism folders) or a sharded format (a folder called ``model_weights``).

With sharded model weights, you can save and load the state of your training script with multiple GPUs or nodes more efficiently and avoid the need to change model partitions when you resume tuning with a different model parallelism setup.

NeMo supports the distributed (sharded) checkpoint format from Megatron Core. Megatron Core supports two checkpoint backends: PyTorch-based (recommended) and Zarr-based (deprecated).
For a detailed explanation check the :doc:`dist_ckpt` guide.


Quantized Checkpoints
---------------------

NeMo provides a :doc:`Post-Training Quantization <../nlp/quantization>` workflow that allows you to convert regular ``.nemo`` models into a `TensorRT-LLM checkpoint <https://nvidia.github.io/TensorRT-LLM/architecture/checkpoint.html>`_, commonly referred to as ``.qnemo`` checkpoints in NeMo. These ``.qnemo`` checkpoints can then be used with the `NVIDIA TensorRT-LLM library <https://nvidia.github.io/TensorRT-LLM/index.html>`_ for efficient inference.

A ``.qnemo`` checkpoint, similar to ``.nemo`` checkpoints, is a tar file that bundles the model configuration specified in the ``config.json`` file along with the ``rank{i}.safetensors`` files. These ``.safetensors`` files store the model weights for each rank individually. In addition, a ``tokenizer_config.yaml`` file is saved, containing only the tokenizer section from the original NeMo ``model_config.yaml`` file. This configuration file defines the tokenizer used by the given model.

When working with large quantized LLMs, it is recommended that you leave the checkpoint uncompressed as a directory rather than a tar file. You can control this behavior by setting the ``compress`` flag when exporting quantized models in `PTQ configuration file <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/conf/megatron_gpt_ptq.yaml>`_.

The following example shows the contents of a quantized model intended to be served using two GPUs (ranks):

.. code-block:: bash

    model-qnemo
    ├── config.json
    ├── rank0.safetensors
    ├── rank1.safetensors
    ├── tokenizer.model
    └── tokenizer_config.yaml

Community Checkpoint Converter
------------------------------
We provide easy-to-use tools that enable users to convert community checkpoints into the NeMo format. These tools facilitate various operations, including resuming training, Supervised Fine-Tuning (SFT), Parameter-Efficient Fine-Tuning (PEFT), and deployment. For detailed instructions and guidelines, please refer to our documentation.

We offer comprehensive guides to assist both end users and developers:

- **User Guide**: Detailed steps on how to convert community model checkpoints for further training or deployment within NeMo. For more information, please see our :doc:`user_guide`.

- **Developer Guide**: Instructions for developers on how to implement converters for community model checkpoints, allowing for broader compatibility and integration within the NeMo ecosystem. For development details, refer to our :doc:`dev_guide`.

- **Megatron-LM Checkpoint Conversion**: NVIDIA NeMo and NVIDIA Megatron-LM share several foundational technologies. You can convert your GPT-style model checkpoints trained with Megatron-LM into the NeMo Framework using our scripts, see our :doc:`convert_mlm`.

.. toctree::
   :maxdepth: 1
   :caption: NeMo Checkpoints

   dist_ckpt
   user_guide
   dev_guide
   convert_mlm
