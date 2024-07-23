Llama 3.1 Law-Domain LoRA Fine-Tuning and Deployment with NeMo Framework and NVIDIA NIM
=======================================================================================

`Llama 3.1 <https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/>`_ are open-source large language models by Meta that deliver state-of-the-art performance on popular industry benchmarks. They have been pretrained on over 15 trillion tokens, and support a 128K token context length. They are available in three sizes, 8B, 70B, and 405B, and each size has two variants—base pretrained and instruction tuned.

`Low-Rank Adaptation (LoRA) <https://arxiv.org/pdf/2106.09685>`__ has emerged as a popular Parameter-Efficient Fine-Tuning (PEFT) technique that tunes a very small number of additional parameters as compared to full fine-tuning, thereby reducing the compute required.

`NVIDIA NeMo
Framework <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>`__ provides tools to perform LoRA on Llama 3.1 to fit your use case, which can then be deployed using `NVIDIA NIM <https://www.nvidia.com/en-us/ai/>`__ for optimized inference on NVIDIA GPUs.

.. figure:: ./img/e2e-lora-train-and-deploy.png
  :width: 1000
  :alt: Diagram showing the steps for LoRA customization using the NVIDIA NeMo Framework and deployment with NVIDIA NIM. The steps include converting the base model to .nemo format, creating LoRA adapters with NeMo, and then depoying the LoRA adapter with NIM for inference.
  :align: center

  Figure 1: Steps for LoRA customization using the NVIDIA NeMo Framework and deployment with NVIDIA NIM


| NIM also enables seamless deployment of multiple LoRA adapters (referred to as “multi-LoRA”) on the same base model. It dynamically loads the adapter weights based on incoming requests at runtime. This flexibility allows handling inputs from various tasks or use cases without deploying a unique model for each individual scenario. For further details, consult the `NIM documentation for LLMs <https://docs.nvidia.com/nim/large-language-models/latest/introduction.html>`__.

Objectives
----------

This tutorial shows how to perform LoRA PEFT on **Llama 3.1 8B Instruct** using a synthetically augmented version of `Law StackExchange <https://huggingface.co/datasets/ymoslem/Law-StackExchange>`__ with NeMo Framework. Law StackExchange is a dataset of legal questions, question titles, and answers. For this demonstration, we will tune the model on the task of title/subject generation, that is, given a Law StackExchange forum question, auto-generate an appropriate title for it. We will then deploy the LoRA tuned model with NVIDIA NIM for inference.

Requirements
-------------

* **Obtain the dataset:** This tutorial is a continuation of the Data Curation tutorial - `Curating Datasets for Parameter Efficient Fine-tuning with Synthetic Data Generation <https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/peft-curation-with-sdg>`__. It demonstrates various filtering and processing operations on the records to improve data quality, as well as (optional) synthetic data generation (SDG) to augment the dataset. Please follow this tutorial to obtain the resulting dataset needed.


* System Configuration
    * Access to at least 1 NVIDIA GPU with a cumulative memory of at least 80GB, for example: 1 x H100-80GB or 1 x A100-80GB.
    * A Docker-enabled environment, with `NVIDIA Container Runtime <https://developer.nvidia.com/container-runtime>`_ installed, which will make the container GPU-aware.
    * `Additional NIM requirements <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#prerequisites>`_.

* `Authenticate with NVIDIA NGC <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-authentication>`_, and download `NGC CLI Tool <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-cli-tool>`_. You will use this tool to download the model and customize it with NeMo Framework.

* Get your Hugging Face `access token <https://huggingface.co/docs/hub/en/security-tokens>`_, which will be used to obtain the tokenizer required during training.


`Process the Dataset with NeMo Curator <https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/peft-curation-with-sdg>`__
-------------------------------------------------------------------------------------------------------

1. Save the dataset in the current directory. You will have obtained `law-qa-{train/val/test}.jsonl` splits resulting from following the abovementioned `data curation tutorial <https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/peft-curation-with-sdg>`__.

.. code:: bash

   mkdir -p curated-data

   # Make sure to update the path below as appropriate
   cp <path/to/generated/data/splits>.jsonl curated-data/.


`Create a LoRA Adapter with NeMo Framework <./llama3-sdg-lora-nemofw.ipynb>`__
------------------------------------------------------------------------------

For LoRA-tuning the model, you will use the NeMo Framework which is available as a `docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`__.


1. Download the `Llama 3.1 8B Instruct .nemo <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/llama-3_1-8b-instruct-nemo>`__ from NVIDIA NGC using the NGC CLI. The following command saves the ``.nemo`` format model in a folder named ``llama-3_1-8b-instruct-nemo_v1.0`` in the current directory. You can specify another path using the ``-d`` option in the CLI tool.

.. code:: bash

   ngc registry model download-version "nvidia/nemo/llama-3_1-8b-instruct-nemo:1.0"



2. Run the container using the following command. It is assumed that you have the dataset, notebook(s), and the `llama-3.1-8b-instruct` model available in the current directory. If not, mount the appropriate folder to ``/workspace``.

.. code:: bash

   export FW_VERSION=24.05.llama3.1


.. code:: bash

   docker run \
     --gpus all \
     --shm-size=2g \
     --net=host \
     --ulimit memlock=-1 \
     --rm -it \
     -v ${PWD}:/workspace \
     -w /workspace \
     -v ${PWD}/results:/results \
     nvcr.io/nvidia/nemo:$FW_VERSION bash

3. From within the container, start the Jupyter lab:

.. code:: bash

   jupyter lab --ip 0.0.0.0 --port=8888 --allow-root

4. Then, navigate to `this notebook <./llama3-sdg-lora-nemofw.ipynb>`__.


`Deploy the LoRA Inference Adapter with NVIDIA NIM <./llama3-sdg-lora-deploy-nim.ipynb>`__
--------------------------------------------------------------------------------------

This procedure demonstrates how to deploy the trained LoRA adapter with NVIDIA NIM. NIM supports LoRA adapters in ``.nemo`` (from NeMo Framework), and Hugging Face model formats. You will deploy the Law StackExchange title-generation LoRA adapter from the first notebook.

1. Prepare the LoRA model store.

After training is complete, that LoRA model checkpoint will be created at ``./results/Meta-llama3.1-8B-Instruct-titlegen/checkpoints/megatron_gpt_peft_lora_tuning.nemo``, assuming default paths in the first notebook weren’t modified.

To ensure the model store is organized as expected, create a folder named ``llama3.1-8b-law-titlegen`` under a model store directory, and move your ``.nemo`` checkpoint there.

.. code:: bash

   # Set path to your LoRA model store
   export LOCAL_PEFT_DIRECTORY="$(pwd)/loras"

   mkdir -p $LOCAL_PEFT_DIRECTORY/llama3.1-8b-law-titlegen

   # Ensure the source path is correct
   cp ./results/Meta-llama3.1-8B-Instruct-titlegen/checkpoints/megatron_gpt_peft_lora_tuning.nemo $LOCAL_PEFT_DIRECTORY/llama3.1-8b-law-titlegen


Ensure that the LoRA model store directory follows this structure: the model name would be name of the sub-folder containing the ``.nemo`` file.

::

   <$LOCAL_PEFT_DIRECTORY>
   └── llama3.1-8b-law-titlegen
       └── megatron_gpt_peft_lora_tuning.nemo


Note that NIM supports deployment of multiple LoRA adapters over the same base model. As such, if you have any other adapters for other tasks trained or available, you can place them in separate sub-folders under `$LOCAL_PEFT_DIRECTORY`.

2. Set-up NIM.

From your host OS environment, start the NIM docker container while mounting the LoRA model store, as follows:

.. code:: bash

   # Set these configurations
   export NGC_API_KEY=<YOUR_NGC_API_KEY>
   export NIM_PEFT_REFRESH_INTERVAL=3600  # (in seconds) will check NIM_PEFT_SOURCE for newly added models in this interval
   export NIM_CACHE_PATH=</path/to/NIM-model-store-cache>  # Model artifacts (in container) are cached in this directory


.. code:: bash

   mkdir -p $NIM_CACHE_PATH
   chmod -R 777 $NIM_CACHE_PATH

   export NIM_PEFT_SOURCE=/home/nvs/loras # Path to LoRA models internal to the container
   export CONTAINER_NAME=meta-llama3.1-8b-instruct

   docker run -it --rm --name=$CONTAINER_NAME \
         --gpus all \
         --network=host \
         --shm-size=16GB \
         -e NGC_API_KEY \
         -e NIM_PEFT_SOURCE \
         -v $NIM_CACHE_PATH:/opt/nim/.cache \
         -v $LOCAL_PEFT_DIRECTORY:$NIM_PEFT_SOURCE \
         nvcr.io/nim/meta/llama-3.1-8b-instruct:1.1.0

The first time you run the command, it will download the model and cache it in ``$NIM_CACHE_PATH`` so subsequent deployments are even faster. There are several options to configure NIM other than the ones listed above. You can find a full list in the `NIM configuration <https://docs.nvidia.com/nim/large-language-models/latest/configuration.html>`__ documentation.


3. Start the notebook.

From another terminal, follow the same instructions as the previous notebook to launch Jupyter Lab, and then navigate to `this notebook <./llama3-sdg-lora-deploy-nim.ipynb>`__.

You can use the same NeMo Framework docker container which has Jupyter Lab already installed.


