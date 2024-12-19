Llama 3 Supervised Fine-Tuning and Parameter Efficient Fine-Tuning with NeMo 2.0
================================================================================

`Llama 3 <https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/>`_ is an open-source large language model by Meta that delivers state-of-the-art performance on popular industry benchmarks. It has been pretrained on over 15 trillion tokens and supports an 8K token context length. It is available in two sizes, 8B and 70B, and each size has two variants—base pretrained and instruction tuned.

Supervised Fine-Tuning (SFT) refers to unfreezing all the weights and layers in our model and training on a newly labeled set of examples. We can fine-tune to incorporate new, domain-specific knowledge, or teach the foundation model what type of response to provide.

`Low-Rank Adaptation (LoRA) <https://arxiv.org/pdf/2106.09685>`__ has emerged as a popular Parameter-Efficient Fine-Tuning (PEFT) technique that tunes a very small number of additional parameters as compared to full fine-tuning, thereby reducing the compute required.

`NVIDIA NeMo
Framework <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>`__ provides tools to perform SFT and LoRA on Llama 3 to fit your use case.

Requirements
------------

* System Configuration
    * For SFT: access to at least 2 NVIDIA GPUs with a cumulative memory of at least 80GB, for example: 2 x H100-80GB or 2 x A100-80GB.
    * For LoRA: access to at least 1 NVIDIA GPUs with a cumulative memory of at least 80GB, for example: 1 x H100-80GB or 1 x A100-80GB.
    * A Docker-enabled environment, with `NVIDIA Container Runtime <https://developer.nvidia.com/container-runtime>`_ installed, which will make the container GPU-aware.
   
* Software Requirements
    * Use the latest [NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) . Note that you must be logged in to the container registry to view this page.
    * This notebook is tested on the container: `nvcr.io/nvidia/nemo:24.12-rc0`.
    * Get your Hugging Face [access token](https://huggingface.co/docs/hub/en/security-tokens), which will be used to obtain the tokenizer required during training.

* NeMo 2.0 and NeMo-Run
    * We will use NeMo 2.0 and NeMo-Run to perform SFT and LoRA on Llama 3. Both are already available in the NeMo Framework Container.


Start the NeMo Framework Container
----------------------------------

1. You can start and enter the dev container by:

.. code:: bash

   docker run \
     --gpus all \
     --shm-size=2g \
     --net=host \
     --ulimit memlock=-1 \
     --rm -it \
     -v ${PWD}:/workspace \
     -w /workspace \
     nvcr.io/nvidia/nemo:24.12-rc0 bash

Once you are inside the container, you can run `nvidia-smi` to verify that the GPUs are accessible.

.. code:: bash

   nvidia-smi


2. You need to request download permission from Meta and Hugging Face. Then, from within the container, log in through `huggingface-cli` using your Hugging Face token. 

.. code:: bash

   huggingface-cli login


3. From within the container, start the Jupyter lab:

.. code:: bash

   jupyter lab --ip 0.0.0.0 --port=8888 --allow-root

4. Then, navigate to `the SFT notebook <./nemo2-sft.ipynb>`__ or `the LoRA notebook <./nemo2-peft.ipynb>`__ to perform SFT or LoRA on Llama 3, respectively.
