# ChipNeMo - Custom tokenization + Domain Adaptive Pre-training on Llama 2 7b with NeMo Framework

[ChipNeMo](https://arxiv.org/pdf/2311.00176) is a chip design domain-adapted Large Language Model (LLM). Instead of directly deploying off-the-shelf commercial or open-source LLMs, the paper adopts the following domain adaptation techniques: domain-adaptive tokenization, domain-adaptive continued pre-training, model alignment with domain-specific instructions, and domain-adapted retrieval models. Specifically, Llama 2 foundation models are continually pre-trained with more than 20 billion tokens on domain-specific chip design data, including code and documents. They are then fine-tuned with instruction datasets from design data as well as external sources. Evaluations on the resultant domain-adapted ChipNeMo model demonstrate that domain-adaptive pre-training of language models can lead to superior performance in domain-related downstream tasks compared to their base Llama 2 counterparts, without degradations in generic capabilities.

Here, we share a tutorial with best practices on custom tokenization and DAPT (Domain-Adaptive Pre-Training) for a ChipNeMo-like code generation use case.

## Requirements

### Software Requirements
* Access to latest NeMo Framework NGC Containers
* This playbook has been tested on: nvcr.io/nvidia/nemo:24.07. It is expected to work similarly on other environments.

### Hardware Requirements
* This playbook can run on CPUs or GPUs. For GPUs, this playbook has been tested on minimum 2xA100 80G

### Data Curation

* In this tutorial, we will leverage chip domain/hardware datasets from open-source GitHub repositories, wiki URLs, and academic papers. Therefore, as a pre-requisite the user should curate the domain specific and general purpose data using the NeMo Curator and place them in the directories mentioned below. 

* `./code/data` should contain curated data from chip domain after processing with NeMo Curator. Playbook for DAPT data curation can be found [here](https://github.com/NVIDIA-NeMo/Curator/tree/dask/tutorials/dapt-curation). Please note that this tutorial uses NeMo Curator version 0.9.0 or lower.

* `./code/general_data` should contain open-source general purpose data that llama-2 was trained on. This data will help idenitfy token/vocabulary differences between general purpose and domain-specific datasets. Data can be downloaded from [Wikepedia](https://huggingface.co/datasets/legacy-datasets/wikipedia), [CommonCrawl](https://data.commoncrawl.org/) etc. and curated with [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator/tree/dask/tutorials/single_node_tutorial). Please note that this tutorial uses NeMo Curator version 0.9.0 or lower.


## Custom Tokenization for DAPT

After placing the curated data in the directories mentioned above, we can proceed with custom tokenization and DAPT. 

* `./code/custom_tokenization.ipynb` walks through the custom tokenization workflow required for DAPT 

## Pretraining for DAPT

Once we have the domain adapted custom tokenizer from above, we can proceed with pretraining using the customer tokenizer.

* `./code/domain_adaptive_pretraining.ipynb` walks through the pretraining workflow required for DAPT 

## Deployment of .nemo via NIMs

Once the domain adapted pretraining is completed a .nemo checkpoint will be saved. Follow the document 'Deploying .nemo as a NIM' to deploy the .nemo checkpoint as NIMs and send inference request. 
