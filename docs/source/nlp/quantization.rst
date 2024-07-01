.. _megatron_quantization:

Quantization
==========================

Post-Training Quantization (PTQ)
--------------------------------

PTQ enables deploying a model in a low-precision format -- FP8, INT4, or INT8 -- for efficient serving. Different quantization methods are available including FP8 quantization, INT8 SmoothQuant, and INT4 AWQ.

Model quantization has two primary benefits: reduced model memory requirements and increased inference throughput.

In NeMo, quantization is enabled by the `NVIDIA TensorRT Model Optimizer (ModelOpt) <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ library -- a library to quantize and compress deep learning models for optimized inference on GPUs.

The quantization process consists of the following steps:

1. Loading a model checkpoint using an appropriate parallelism strategy
2. Calibrating the model to obtain appropriate algorithm-specific scaling factors
3. Producing an output directory or .qnemo tarball with model config (json), quantized weights (safetensors) and tokenizer config (yaml).

Loading models requires using an ModelOpt spec defined in `nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron/gpt_layer_modelopt_spec.py>`_ module. Typically the calibration step is lightweight and uses a small dataset to obtain appropriate statistics for scaling tensors. The output directory produced (or a .qnemo tarball) is ready to be used to build a serving engine with the Nvidia TensorRT-LLM library. The engine build step is also available in NeMo project in ``nemo.deploy`` and ``nemo.export`` modules.

Quantization algorithm can also be conveniently set to ``"null"`` to perform only the weights export step using default precision for TensorRT-LLM deployment. This is useful to obtain baseline performance and accuracy results for comparison.

Support Matrix
^^^^^^^^^^^^^^

Table below presents verified model support matrix for popular LLM architectures. Each model entry also optionally provides a download link to a corresponding Nemo checkpoint for testing purposes. Support for other model families is experimental.

.. list-table:: Model Support Matrix
   :widths: 15 15 15 15
   :header-rows: 1

   * - **Model Family**
     - **FP8**
     - **INT8_SQ**
     - **INT4_AWQ**
   * - Llama (1, 2, 3)
     - ✅
     - ✅
     - ✅
   * - Mistral
     - ✅
     - ✅
     - ✅
   * - `GPT-3 <https://huggingface.co/nvidia/GPT-2B-001>`_
     - ✅
     - ✅
     - ✅
   * - `Nemotron-3 8b <https://huggingface.co/nvidia/nemotron-3-8b-base-4k>`_
     - ✅
     - ✅
     - ✅
   * - Nemotron-4 15b
     - ✅
     - ✅
     - ✅
   * - `Nemotron-4 340b <https://huggingface.co/nvidia/Nemotron-4-340B-Base>`_  (Base, Instruct, Reward)
     - ✅
     - ✅
     - ✅
   * - StarCoder 2
     - ✅
     - ✅
     - ✅
   * - Gemma
     - ✅
     - ✅
     - ✅


Example
^^^^^^^
The example below shows how to quantize the Llama3 70b model into FP8 precision, using tensor parallelism of 8 on a single DGX H100 node. The quantized model is designed for serving using 2 GPUs specified with the ``export.inference_tensor_parallel`` parameter.

The script must be launched correctly with the number of processes equal to tensor parallelism. This is achieved with the ``torchrun`` command below:

.. code-block:: bash

    torchrun --nproc-per-node 8 examples/nlp/language_modeling/megatron_gpt_ptq.py \
        model.restore_from_path=llama3-70b-base-bf16.nemo \
        model.tensor_model_parallel_size=8 \
        model.pipeline_model_parallel_size=1 \
        trainer.num_nodes=1 \
        trainer.devices=8 \
        trainer.precision=bf16 \
        quantization.algorithm=fp8 \
        export.decoder_type=llama \
        export.inference_tensor_parallel=2 \
        export.save_path=llama3-70b-base-fp8-qnemo

For large models, the command can be used in multi-node setting. For example, this can be done with `NeMo Framework Launcher <https://github.com/NVIDIA/NeMo-Framework-Launcher>`_ using Slurm.

The output directory stores the following files:

.. code-block:: bash

    llama3-70b-base-fp8-qnemo/
    ├── config.json
    ├── rank0.safetensors
    ├── rank1.safetensors
    ├── tokenizer.model
    └── tokenizer_config.yaml


The TensorRT-LLM engine can be conveniently built and run using ``TensorRTLLM`` class available in ``nemo.export`` submodule:

.. code-block:: python

    from nemo.export.tensorrt_llm import TensorRTLLM


    trt_llm_exporter = TensorRTLLM(model_dir="/path/to/trt_llm_engine_folder")
    trt_llm_exporter.export(
        nemo_checkpoint_path="llama3-70b-base-fp8-qnemo",
        model_type="llama",
    )
    trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])


Alternatively, it can also be built directly using ``trtllm-build`` command, see `TensorRT-LLM documentation <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#fp8-post-training-quantization>`_:

.. code-block:: bash

    trtllm-build \
        --checkpoint_dir llama3-70b-base-fp8-qnemo \
        --output_dir /path/to/trt_llm_engine_folder \
        --max_batch_size 8 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --strongly_typed


Known issues
^^^^^^^^^^^^
* Currently with ``nemo.export`` module building TensorRT-LLM engines for quantized "qnemo" models is limited to single-node deployments.


Quantization-Aware Training (QAT)
---------------------------------

QAT is the technique of fine-tuning a quantized model to recover model quality degradation due to quantization.
During QAT, the quantization scaling factors computed during PTQ are frozen and the model weights are fine-tuned.
While QAT requires much more compute resources than PTQ, it is highly effective in recovering model quality.
To perform QAT on a calibrated model from PTQ, you need to further fine-tune the model on a downstream task using a small dataset before exporting to TensorRT-LLM.
You can reuse your training pipeline for QAT.
As a rule of thumb, we recommend QAT for 1-10% original training duration and a small learning rate, e.g. 1e-5 for Adam optimizer.
If you are doing QAT on an SFT model where learning rates and finetuning dataset size are already small, you can continue using the same SFT learning rate and dataset size as a starting point for QAT.
Since QAT is done after PTQ, the supported model families are the same as for PTQ.


Example
^^^^^^^

The example below shows how to perform PTQ and QAT on a Supervised Finetuned Llama2 7B model to INT4 precision.
The script is tested using tensor parallelism of 8 on 8x RTX 6000 Ada 48GB GPUs. Alternatively, a single DGX A100 node with 8x 40GB GPUs can be used for the same purpose.
For bigger models like Llama2 70B, you may need to use one or more DGX H100 nodes with 8x 80GB GPUs each.

The example is a modified version of the `SFT with Llama 2 playbook <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2sft.html>`_.
Please refer to the playbook for more details on setting up a BF16 NeMo model and the ``databricks-dolly-15k`` instruction dataset.

First we will run the SFT example command from the playbook as-is to train a Llama2 7B SFT model for 100 steps.
Make sure to change ``trainer.max_steps=50`` to ``trainer.max_steps=100`` for the ``examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py`` script.
This will take ~2 hours to produce a model checkpoint with validation loss approximately ``1.15`` that we will use for PTQ and QAT next.

For Quantization, we use a modified version of the sft script and config file which includes the quantization and TensorRT-LLM export support.
Along with the new parameters, make sure to pass the same parameters you passed for SFT training except the model restore path will be the SFT output ``.nemo`` file.
The below example command will perform PTQ on the SFT model checkpoint followed by SFT again (QAT) which can then be exported for TensorRT-LLM inference. The script will take ~2-3 hours to complete.

.. code-block:: bash

    torchrun --nproc-per-node 8 examples/nlp/language_modeling/tuning/megatron_gpt_qat.py \
        trainer.num_nodes=1 \
        trainer.devices=8 \
        trainer.precision=bf16 \
        trainer.max_steps=100 \
        model.restore_from_path=<llama2-7b-sft-nemo-path> \
        model.global_batch_size=128 \
        quantization.algorithm=int4 \
        # other parameters from sft training

As you can see from the logs, the INT4 PTQ model has a validation loss of approximately ``1.31`` and the QAT model has a validation loss of approximately ``1.17`` which is very close to the BF16 model loss of ``1.15``.
This script will produce a quantized ``.nemo`` checkpoint at the experiment manager log directory (in the config yaml file) that can be used for further training.
It can also optionally produce an exported TensorRT-LLM engine directory or a ``.qnemo`` file that can be used for inference by setting the ``export`` parameters similar to the PTQ example.
Note that you may tweak the QAT trainer steps and learning rate if needed to achieve better model quality.


References
----------

Please refer to the following papers for more details on quantization techniques:

* `Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation, 2020 <https://arxiv.org/abs/2004.09602>`_
* `FP8 Formats for Deep Learning, 2022 <https://arxiv.org/abs/2209.05433>`_
* `SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, 2022 <https://arxiv.org/abs/2211.10438>`_
* `AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, 2023 <https://arxiv.org/abs/2306.00978>`_
