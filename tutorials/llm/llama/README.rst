
Getting Started with Llama 3 and Llama 3.1
==========================================

.. note::
   **Deprecation Notice:** NeMo 2.0 LLM/VLM support is deprecated and will be replaced by 
   `NeMo Megatron-Bridge <https://github.com/NVIDIA-NeMo/Megatron-Bridge>`_ and 
   `NeMo AutoModel <https://github.com/NVIDIA-NeMo/AutoModel>`_. 
   Please refer to those projects for LLM training and fine-tuning workflows.

This repository contains Jupyter Notebook tutorials using the NeMo Framework for Llama-3 and Llama-3.1 models by Meta.

.. list-table:: 
   :widths: 100 25 100
   :header-rows: 1

   * - Tutorial
     - Dataset
     - Description
   * - `Llama 3 LoRA Fine-Tuning and Multi-LoRA Deployment with NeMo Framework and NVIDIA NIM <./biomedical-qa>`_
     - `PubMedQA <https://pubmedqa.github.io/>`_
     - Perform LoRA PEFT on Llama 3 8B Instruct using a dataset for bio-medical domain question answering. Deploy multiple LoRA adapters with NVIDIA NIM.
   * - `Llama 3.1 Law-Domain LoRA Fine-Tuning and Deployment with NeMo Framework and NVIDIA NIM <./sdg-law-title-generation>`_
     - `Law StackExchange <https://huggingface.co/datasets/ymoslem/Law-StackExchange>`_
     - Perform LoRA PEFT on Llama 3.1 8B Instruct using a synthetically augmented version of Law StackExchange with NeMo Framework, followed by deployment with NVIDIA NIM. As a prerequisite, follow the tutorial for `data curation using NeMo Curator <https://github.com/NVIDIA-NeMo/Curator/tree/dask/tutorials/peft-curation-with-sdg>`_.
   * - `Llama 3.1 Pruning and Distillation with NeMo Framework <./pruning-distillation>`_
     - `WikiText-103-v1 <https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1>`_
     - Perform pruning and distillation on Llama 3.1 8B using the WikiText-103-v1 dataset with NeMo Framework.
   * - `Llama3 LoRA Fine-Tuning and Supervised Fine-Tuning using NeMo2 <./nemo2-sft-peft>`_
     - `SQuAD <https://arxiv.org/abs/1606.05250>`_ for LoRA and `Databricks-dolly-15k <https://huggingface.co/datasets/databricks/databricks-dolly-15k>`_ for SFT
     - Perform LoRA PEFT and SFT on Llama 3 8B using NeMo 2.0
   * - `NeMo SlimPajama Data Pipeline and Pretraining tutorial <./slimpajama>`_
     - `SlimPajama <https://huggingface.co/datasets/cerebras/SlimPajama-627B>`_
     - Step-by-step guide to preprocess the SlimPajama dataset and pretrain a Llama3 8B model using NeMo 2.0
