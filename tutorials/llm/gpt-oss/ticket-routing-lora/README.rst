Fine-tuning OpenAI gpt-oss with NVIDIA NeMo Framework
=====================================================

`GPT-OSS <https://blogs.nvidia.com/blog/openai-gpt-oss/>`_ are open-weights large language models by OpenAI that deliver state-of-the-art performance on a variety of tasks including reasoning and tool use. They are available in two sizes `gpt-oss-120b` and `gpt-oss-20b`. Both models employ a mixture-of-experts (MoE) transformer architecture enabling them with a larger parameter count to tackle complex tasks while running efficiently on modern GPUs.

Supervised Finetuning (SFT) and `Low-Rank Adaptation (LoRA) <https://arxiv.org/pdf/2106.09685>`__ are popular techniques to adapt models for specific usecases. `NVIDIA NeMo
Framework <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>`__ provides tools to perform SFT and/or LoRA on gpt-oss models to fit your use case.

For more comprehensive information on the NeMo framework's support for gpt-oss, please refer to the `official documentation <https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html>`_.

Objectives
----------

Customer service is a key use case for many enterprises today. It's an area where AI agents have shown early signs of positive business outcomes. This tutorial demonstrates how to perform LoRA PEFT on `gpt-oss-20b <https://huggingface.co/openai/gpt-oss-20b>`_ for a simple customer-service ticket-routing use case using the NeMo framework. The `multilingual-customer-support-tickets <https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets>`_ dataset on Kaggle contains labeled email tickets, including agent responses, priority, and queue. We use this dataset to fine-tune the model for queue prediction.

Requirements
-------------

* System Configuration
    * Access to at least 1 NVIDIA GPU with a cumulative memory of at least 80GB, for example: 1 x H100-80GB or 1 x A100-80GB.
    * A Docker-enabled environment, with `NVIDIA Container Runtime <https://developer.nvidia.com/container-runtime>`_ installed, which will make the container GPU-aware.


* Get your Hugging Face `access token <https://huggingface.co/docs/hub/en/security-tokens>`_, which will be used to obtain certain artifacts required during training.

* Log into `Kaggle <https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets>`_, where you will download the dataset from.


`Obtain the dataset <https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets>`_
---------------------------------------------------------------------------------------------------------

1. Download the dataset from Kaggle.

You may manually download the dataset as a .zip from Kaggle by using the Download button on top right corner. Alternatively, you can use the `Kaggle CLI <https://github.com/Kaggle/kaggle-api>`_.

.. code:: bash

   mkdir -p nemo-experiments/data

   # Option 1: Download the data manually from Kaggle website.

   # Option 2: Download the dataset from Kaggle CLI
   # kaggle datasets download tobiasbueck/multilingual-customer-support-tickets

   # Unzip the downloaded file
   unzip archive.zip -d nemo-experiments/data/customer-ticket-routing

Ensure your data is in the ``nemo-experiments/data`` folder before proceeding.


`LoRA with NeMo Framework <./gpt-oss-lora.ipynb>`_
--------------------------------------------------

For LoRA-tuning the model, you will use the NeMo Framework which is available as a `docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`__.


1. Obtain the model (This only needs to be done once)

.. code:: bash

   mkdir -p nemo-experiments/models && cd nemo-experiments/models

   # Install git-lfs if not already installed
   apt-get update && apt-get install git-lfs
   git lfs install

   # Clone the model repo
   git clone https://huggingface.co/openai/gpt-oss-20b



2. Run the container using the following command. It is assumed that you the notebook available in the current directory. If not, mount the appropriate folder to ``/workspace``. All the model artifacts will be downloaded to the ``nemo-experiments`` folder, which you can configure as needed.

.. code:: bash

   export FW_VERSION=25.07.gpt_oss


.. code:: bash

   docker run \
     --gpus all \
     --shm-size=8g \
     --net=host \
     --ulimit memlock=-1 \
     --rm -it \
     -v ${PWD}:/workspace \
     -w /workspace \
     -v ${PWD}/nemo-experiments:/nemo-experiments \
     nvcr.io/nvidia/nemo:$FW_VERSION bash


3. From within the container, start the Jupyter lab:

.. code:: bash

   jupyter lab --ip 0.0.0.0 --port=8888 --allow-root


4. Then, navigate to `this notebook <./gpt-oss-lora.ipynb>`__. The result of this notebook will be a .safetensors formatted finetuned full-weights checkpoint.