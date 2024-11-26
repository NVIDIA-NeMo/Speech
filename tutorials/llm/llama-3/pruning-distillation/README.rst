Llama 3.1 Pruning and Distillation with NeMo Framework
=======================================================================================

`Llama 3.1 <https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/>`_ models, developed by Meta, are open-source large language models that deliver state-of-the-art performance on popular industry benchmarks. Pretrained on over 15 trillion tokens, they support a 128K token context length. These models are available in three sizes: 8B, 70B, and 405B. Each size offers two variants: base pretrained and instruction tuned.

`NVIDIA NeMo Framework <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>`_ provides tools to perform teacher fine-tuning, pruning, and distillation on Llama 3.1 to fit your use case.

`NVIDIA TensorRT Model Optimizer <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ is a library (referred to as **Model Optimizer**, or **ModelOpt**) comprising state-of-the-art model optimization techniques including `quantization <https://github.com/NVIDIA/TensorRT-Model-Optimizer#quantization>`_, `sparsity <https://github.com/NVIDIA/TensorRT-Model-Optimizer#sparsity>`_, `distillation <https://github.com/NVIDIA/TensorRT-Model-Optimizer#distillation>`_, and `pruning <https://github.com/NVIDIA/TensorRT-Model-Optimizer#pruning>`_ to compress models.

`LLM Pruning and Distillation in Practice: The Minitron Approach <https://arxiv.org/abs/2408.11796>`_ provides tools to perform teacher fine-tuning, pruning, and distillation on Llama 3.1 as described in the `tech report <https://arxiv.org/abs/2408.11796>`_.

`How to Prune and Distill Llama-3.1 8B to an NVIDIA Llama-3.1-Minitron 4B Model <https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/>`_ provides practical and effective structured compression best practices for LLMs that combine depth, width, attention, and MLP pruning with knowledge distillation-based retraining. These strategies are presented in the `Compact Language Models via Pruning and Knowledge Distillation <https://arxiv.org/pdf/2407.14679>`_ paper.

`Mistral-NeMo-Minitron 8B Model Delivers Unparalleled Accuracy <https://developer.nvidia.com/blog/mistral-nemo-minitron-8b-foundation-model-delivers-unparalleled-accuracy/>`_ introduces the Mistral-NeMo-Minitron 8B, a state-of-the-art 8 billion parameter language model created by pruning and distilling the larger Mistral NeMo 12B model.

Objectives
----------

This tutorial demonstrates how to perform depth-pruning, width-pruning, teacher fine-tuning, and distillation on **Llama 3.1 8B** using the `WikiText-103-v1 <https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1>`_ dataset with the NeMo Framework. The `WikiText-103-v1 <https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1>`_ language modeling dataset comprises over 100 million tokens extracted from verified Good and Featured articles on Wikipedia.

For this demonstration, we will perform teacher correction by running a light fine-tuning procedure on the ``Meta LLama 3.1 8B`` teacher model to generate a fine-tuned teacher model, ``megatron_llama_ft.nemo``, needed for optimal distillation. This fine-tuned teacher model is then trimmed. There are two methods to prune a model: depth-pruning and width-pruning. We will explore both techniques, yielding ``4b_depth_pruned_model.nemo`` and ``4b_width_pruned_model.nemo``, respectively. These models will serve as starting points for distillation to create the final distilled 4B models.

We are using models utilizing the ``meta-llama/Meta-Llama-3.1-8B`` tokenizer for this demonstration.

``NOTE:`` A subset of functions is being demonstrated in the notebooks. Some features like Neural Architecture Search (NAS) are unavailable, but will be supported in future releases.

Requirements
-------------

* System Configuration
    * Access to at least 8 NVIDIA GPUs, each with a memory of at least 80GB (e.g., 8 x H100-80GB or 8 x A100-80GB).
    * A Docker-enabled environment, with `NVIDIA Container Runtime <https://developer.nvidia.com/container-runtime>`_ installed, which will make the container GPU-aware.

* `Authenticate with NVIDIA NGC <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-authentication>`_ and download `NGC CLI Tool <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-cli-tool>`_. You will use this tool to download the model and customize it with NeMo Framework.

* Get your Hugging Face `access token <https://huggingface.co/docs/hub/en/security-tokens>`_, which will be used to obtain the tokenizer required during training.

``NOTE:`` The default configuration in the notebook runs on 8 x 80GB NVIDIA GPUs. However, you can potentially reduce the Tensor Parallel size ``(TENSOR_PARALLEL_SIZE)`` along with the Micro-Batchsize ``(MICRO_BATCH_SIZE)`` in the teacher fine-tuning and distillation scripts to accommodate lower resource availability.

Create a Pruned and Distilled Model with NeMo Framework
------------------------------------------------------------------------------

For pruning and distilling the model, you will use the NeMo Framework, which is available as a `Docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`_.

``NOTE:`` These notebooks use the `NVIDIA TensorRT Model Optimizer <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ under the hood for pruning and distillation.


1. Download the `Llama 3.1 8B .nemo <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/llama-3_1-8b-nemo>`_ from NVIDIA NGC using the `NGC CLI <https://org.ngc.nvidia.com/setup/installers/cli>`_. Generate the ``NGC_API_KEY`` following these `instructions <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#option-2-from-ngc>`_. The following command saves the ``.nemo`` format model in a folder named ``llama-3_1-8b-nemo_v1.0`` in the current directory. You can specify another path using the ``-d`` option in the CLI tool.

.. code:: bash

   ngc registry model download-version "nvidia/nemo/llama-3_1-8b-nemo:1.0"

2. Run the container using the following command. It is assumed that you have the dataset, notebook(s), and the ``llama3_1_8b.nemo`` model available in the current directory. If not, mount the appropriate folder to ``/workspace``.

.. code:: bash

   export FW_VERSION=dev


.. code:: bash

   docker run \
     --gpus all \
     --shm-size=16GB \
     --net=host \
     --ulimit memlock=-1 \
     --rm -it \
     -v ${PWD}:/workspace \
     -w /workspace \
     nvcr.io/nvidia/nemo:$FW_VERSION bash

3. From within the container, start the Jupyter lab:

.. code:: bash

   jupyter lab --ip 0.0.0.0 --port=8888 --allow-root

4. Then, navigate to `this notebook <./introduction.ipynb>`_ to get started.

This directory contains a list of notebooks that cover all the steps to create a distilled 4B model.

:: 

   <$pruning_distillation>
   └── introduction.ipynb
   └── 01_data_preparation.ipynb
   └── 02_teacher_finetuning.ipynb
   └── 03_a_depth_pruning.ipynb
   └── 03_b_width_pruning.ipynb
   └── 04_a_distilling_depth_pruned_student.ipynb
   └── 04_b_distilling_width_pruned_student.ipynb
   └── 05_display_results.ipynb
   
Results
------------------------------------------------------------------------------
``NOTE:`` This notebook demonstrates the use of the teacher fine-tuning, pruning, and the distillation scripts. These scripts should ideally be run on a multi-node cluster with a larger ``GLOBAL_BATCH_SIZE`` and ``STEPS`` to see improvement in the validation loss.

Here are the validation loss plots over 30 steps of running the training step in the distillation script (at the end of the `notebook <./05_display_results.ipynb>`_).

.. figure:: https://github.com/NVIDIA/NeMo/releases/download/r2.0.0rc1/val_loss_depth_pruned_student_distillation.png
  :width: 400px
  :alt: Diagram showing the validation loss over 30 steps of running the training step in the distillation script when using the depth-pruned model as the student
  :align: center

  Figure 1: Validation Loss Plot When Using the Depth-Pruned Model as the Student
  
.. figure:: https://github.com/NVIDIA/NeMo/releases/download/r2.0.0rc1/val_loss_width_pruned_student_distillation.png
  :width: 400px
  :alt: Diagram showing the validation loss over 30 steps of running the training step in the distillation script when using the width-pruned model as the student
  :align: center

  Figure 2: Validation Loss Plot When Using the Width-Pruned Model as the Student