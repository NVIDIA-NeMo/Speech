
Getting Started with Llama 3 and Llama 3.1
==========================================

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
     - Perform LoRA PEFT on Llama 3.1 8B Instruct using a synthetically augmented version of Law StackExchange with NeMo Framework, followed by deployment with NVIDIA NIM. As a prerequisite, follow the tutorial for `data curation using NeMo Curator <https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/peft-curation-with-sdg>`_.
   * - `Llama 3.1 Pruning and Distillation with NeMo Framework <./pruning-distillation>`_
     - `WikiText-103-v1 <https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1>`_
     - Perform pruning and distillation on Llama 3.1 8B using the WikiText-103-v1 dataset with NeMo Framework.
