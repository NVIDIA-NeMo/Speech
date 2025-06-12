[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# **NVIDIA NeMo Framework**

## Latest News

<!-- markdownlint-disable -->
<details open>
  <summary><b>Pretrain and finetune :hugs:Hugging Face models via AutoModel</b></summary>
      Nemo Framework's latest feature AutoModel enables broad support for :hugs:Hugging Face models, with 25.04 focusing on

  
- <a href=https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#automodelforcausallm>AutoModelForCausalLM<a> in the <a href="https://huggingface.co/models?pipeline_tag=text-generation&sort=trending">Text Generation<a> category
- <a href=https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForImageTextToText>AutoModelForImageTextToText<a> in the <a href="https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending">Image-Text-to-Text<a> category

More Details in Blog: <a href=https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework>Run Hugging Face Models Instantly with Day-0 Support from NVIDIA NeMo Framework<a>. Future releases will enable support for more model families such as Video Generation models.(2025-05-19)
</details>

<details open>
  <summary><b>Training on Blackwell using Nemo</b></summary>
      NeMo Framework has added Blackwell support, with <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html>performance benchmarks on GB200 & B200<a>. More optimizations to come in the upcoming releases.(2025-05-19)
</details>

<details open>
  <summary><b>Training Performance on GPU Tuning Guide</b></summary>
      NeMo Framework has published <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html>a comprehensive guide for performance tuning to achieve optimal throughput<a>! (2025-05-19)
</details>

<details open>
  <summary><b>New Models Support</b></summary>
      NeMo Framework has added support for latest community models - <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html>Llama 4<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vision/diffusionmodels/flux.html>Flux<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama_nemotron.html>Llama Nemotron<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/hyena.html#>Hyena & Evo2<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/qwen2vl.html>Qwen2-VL<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/qwen2.html>Qwen2.5<a>, Gemma3, Qwen3-30B&32B.(2025-05-19)
</details>


<details open>
  <summary><b>NeMo Framework 2.0</b></summary>
      We've released NeMo 2.0, an update on the NeMo Framework which prioritizes modularity and ease-of-use. Please refer to the <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html>NeMo Framework User Guide</a> to get started.
</details>
<details open>
  <summary><b>New Cosmos World Foundation Models Support</b></summary>
    <details> 
      <summary> <a href="https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform">Advancing Physical AI with NVIDIA Cosmos World Foundation Model Platform </a> (2025-01-09) 
      </summary> 
        The end-to-end NVIDIA Cosmos platform accelerates world model development for physical AI systems. Built on CUDA, Cosmos combines state-of-the-art world foundation models, video tokenizers, and AI-accelerated data processing pipelines. Developers can accelerate world model development by fine-tuning Cosmos world foundation models or building new ones from the ground up. These models create realistic synthetic videos of environments and interactions, providing a scalable foundation for training complex systems, from simulating humanoid robots performing advanced actions to developing end-to-end autonomous driving models. 
        <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/">
          Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities
        </a> (2025-01-07)
      </summary>
        The NeMo Framework now supports training and customizing the <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> collection of world foundation models. Cosmos leverages advanced text-to-world generation techniques to create fluid, coherent video content from natural language prompts.
        <br><br>
        You can also now accelerate your video processing step using the <a href="https://developer.nvidia.com/nemo-curator-video-processing-early-access">NeMo Curator</a> library, which provides optimized video processing and captioning features that can deliver up to 89x faster video processing when compared to an unoptimized CPU pipeline.
      <br><br>
    </details>
</details>
<details open>
  <summary><b>Large Language Models and Multimodal Models</b></summary>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/state-of-the-art-multimodal-generative-ai-model-development-with-nvidia-nemo/">
          State-of-the-Art Multimodal Generative AI Model Development with NVIDIA NeMo
        </a> (2024-11-06)
      </summary>
        NVIDIA recently announced significant enhancements to the NeMo platform, focusing on multimodal generative AI models. The update includes NeMo Curator and the Cosmos tokenizer, which streamline the data curation process and enhance the quality of visual data. These tools are designed to handle large-scale data efficiently, making it easier to develop high-quality AI models for various applications, including robotics and autonomous driving. The Cosmos tokenizers, in particular, efficiently map visual data into compact, semantic tokens, which is crucial for training large-scale generative models. The tokenizer is available now on the <a href=http://github.com/NVIDIA/cosmos-tokenizer/NVIDIA/cosmos-tokenizer>NVIDIA/cosmos-tokenizer</a> GitHub repo and on <a href=https://huggingface.co/nvidia/Cosmos-Tokenizer-CV8x8x8>Hugging Face</a>.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama/index.html#new-llama-3-1-support for more information/">
        New Llama 3.1 Support
        </a> (2024-07-23)
      </summary>
        The NeMo Framework now supports training and customizing the Llama 3.1 collection of LLMs from Meta.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://aws.amazon.com/blogs/machine-learning/accelerate-your-generative-ai-distributed-training-workloads-with-the-nvidia-nemo-framework-on-amazon-eks/">
          Accelerate your Generative AI Distributed Training Workloads with the NVIDIA NeMo Framework on Amazon EKS
        </a> (2024-07-16)
      </summary>
     NVIDIA NeMo Framework now runs distributed training workloads on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster. For step-by-step instructions on creating an EKS cluster and running distributed training workloads with NeMo, see the GitHub repository <a href="https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/2.nemo-launcher/EKS/"> here.</a>
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/">
          NVIDIA NeMo Accelerates LLM Innovation with Hybrid State Space Model Support
        </a> (2024/06/17)
      </summary>
     NVIDIA NeMo and Megatron Core now support pre-training and fine-tuning of state space models (SSMs). NeMo also supports training models based on the Griffin architecture as described by Google DeepMind. 
      <br><br>
    </details>
      <details>
      <summary>
        <a href="https://huggingface.co/models?sort=trending&search=nvidia%2Fnemotron-4-340B">
          NVIDIA releases 340B base, instruct, and reward models pretrained on a total of 9T tokens.
        </a> (2024-06-18)
      </summary>
      See documentation and tutorials for SFT, PEFT, and PTQ with 
      <a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/nemotron/index.html">
        Nemotron 340B 
      </a>
      in the NeMo Framework User Guide.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/">
          NVIDIA sets new generative AI performance and scale records in MLPerf Training v4.0
        </a> (2024/06/12)
      </summary>
      Using NVIDIA NeMo Framework and NVIDIA Hopper GPUs NVIDIA was able to scale to 11,616 H100 GPUs and achieve near-linear performance scaling on LLM pretraining. 
      NVIDIA also achieved the highest LLM fine-tuning performance and raised the bar for text-to-image training.
      <br><br>
    </details>
    <details>
        <summary>
          <a href="https://cloud.google.com/blog/products/compute/gke-and-nvidia-nemo-framework-to-train-generative-ai-models">
            Accelerate your generative AI journey with NVIDIA NeMo Framework on GKE
          </a> (2024/03/16)
        </summary>
        An end-to-end walkthrough to train generative AI models on the Google Kubernetes Engine (GKE) using the NVIDIA NeMo Framework is available at https://github.com/GoogleCloudPlatform/nvidia-nemo-on-gke. 
        The walkthrough includes detailed instructions on how to set up a Google Cloud Project and pre-train a GPT model using the NeMo Framework.
        <br><br>
      </details>
</details>
<details open>
  <summary><b>Speech Recognition</b></summary>
  <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/">
          Accelerating Leaderboard-Topping ASR Models 10x with NVIDIA NeMo
        </a> (2024/09/24)
      </summary>
      NVIDIA NeMo team released a number of inference optimizations for CTC, RNN-T, and TDT models that resulted in up to 10x inference speed-up. 
      These models now exceed an inverse real-time factor (RTFx) of 2,000, with some reaching RTFx of even 6,000.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/">
          New Standard for Speech Recognition and Translation from the NVIDIA NeMo Canary Model
        </a> (2024/04/18)
      </summary>
      The NeMo team just released Canary, a multilingual model that transcribes speech in English, Spanish, German, and French with punctuation and capitalization. 
      Canary also provides bi-directional translation, between English and the three other supported languages.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/">
          Pushing the Boundaries of Speech Recognition with NVIDIA NeMo Parakeet ASR Models
        </a> (2024/04/18)
      </summary>
      NVIDIA NeMo, an end-to-end platform for the development of multimodal generative AI models at scale anywhere—on any cloud and on-premises—released the Parakeet family of automatic speech recognition (ASR) models. 
      These state-of-the-art ASR models, developed in collaboration with Suno.ai, transcribe spoken English with exceptional accuracy.
      <br><br>
    </details>
  <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/">
        Turbocharge ASR Accuracy and Speed with NVIDIA NeMo Parakeet-TDT
      </a> (2024/04/18)
    </summary>
    NVIDIA NeMo, an end-to-end platform for developing multimodal generative AI models at scale anywhere—on any cloud and on-premises—recently released Parakeet-TDT. 
    This new addition to the  NeMo ASR Parakeet model family boasts better accuracy and 64% greater speed over the previously best model, Parakeet-RNNT-1.1B.
    <br><br>
  </details>
</details>
<!-- markdownlint-enable -->

## Introduction

NVIDIA NeMo Framework is a scalable and cloud-native generative AI
framework built for researchers and PyTorch developers working on Large
Language Models (LLMs), Multimodal Models (MMs), Automatic Speech
Recognition (ASR), Text to Speech (TTS), and Computer Vision (CV)
domains. It is designed to help you efficiently create, customize, and
deploy new generative AI models by leveraging existing code and
pre-trained model checkpoints.

For technical documentation, please see the [NeMo Framework User
Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## What's New in NeMo 2.0

NVIDIA NeMo 2.0 introduces several significant improvements over its predecessor, NeMo 1.0, enhancing flexibility, performance, and scalability.

- **Python-Based Configuration** - NeMo 2.0 transitions from YAML files to a Python-based configuration, providing more flexibility and control. This shift makes it easier to extend and customize configurations programmatically.

- **Modular Abstractions** - By adopting PyTorch Lightning’s modular abstractions, NeMo 2.0 simplifies adaptation and experimentation. This modular approach allows developers to more easily modify and experiment with different components of their models.

- **Scalability** - NeMo 2.0 seamlessly scaling large-scale experiments across thousands of GPUs using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run), a powerful tool designed to streamline the configuration, execution, and management of machine learning experiments across computing environments.

Overall, these enhancements make NeMo 2.0 a powerful, scalable, and user-friendly framework for AI model development.

> [!IMPORTANT]  
> NeMo 2.0 is currently supported by the LLM (large language model) and VLM (vision language model) collections.

### Get Started with NeMo 2.0

- Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
- For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
- [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
- For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
- To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

### Get Started with Cosmos

NeMo Curator and NeMo Framework support video curation and post-training of the Cosmos World Foundation Models, which are open and available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) and [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6). For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## LLMs and MMs Training, Alignment, and Customization

All NeMo models are trained with
[Lightning](https://github.com/Lightning-AI/lightning). Training is
automatically scalable to 1000s of GPUs. You can check the performance benchmarks using the
latest NeMo Framework container [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

When applicable, NeMo models leverage cutting-edge distributed training
techniques, incorporating [parallelism
strategies](https://docs.nvidia.com/nemo-framework/user-guide/latest/modeloverview.html)
to enable efficient training of very large models. These techniques
include Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully
Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and Mixed
Precision Training with BFloat16 and FP8, as well as others.

NeMo Transformer-based LLMs and MMs utilize [NVIDIA Transformer
Engine](https://github.com/NVIDIA/TransformerEngine) for FP8 training on
NVIDIA Hopper GPUs, while leveraging [NVIDIA Megatron
Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for
scaling Transformer model training.

NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM,
Direct Preference Optimization (DPO), and Reinforcement Learning from
Human Feedback (RLHF). See [NVIDIA NeMo
Aligner](https://github.com/NVIDIA/NeMo-Aligner) for more information.

In addition to supervised fine-tuning (SFT), NeMo also supports the
latest parameter efficient fine-tuning (PEFT) techniques such as LoRA,
P-Tuning, Adapters, and IA3. Refer to the [NeMo Framework User
Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html)
for the full list of supported models and techniques.

## LLMs and MMs Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized with [NVIDIA NeMo
Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for
production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## NeMo Framework Launcher

> [!IMPORTANT]  
> NeMo Framework Launcher is compatible with NeMo version 1.0 only. [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) is recommended for launching experiments using NeMo 2.0.

[NeMo Framework
Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) is a
cloud-native tool that streamlines the NeMo Framework experience. It is
used for launching end-to-end NeMo Framework training jobs on CSPs and
Slurm clusters.

The NeMo Framework Launcher includes extensive recipes, scripts,
utilities, and documentation for training NeMo LLMs. It also includes
the NeMo Framework [Autoconfigurator](https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration),
which is designed to find the optimal model parallel configuration for
training on a specific cluster.

To get started quickly with the NeMo Framework Launcher, please see the
[NeMo Framework
Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).
The NeMo Framework Launcher does not currently support ASR and TTS
training, but it will soon.

## Get Started with NeMo Framework

Getting started with NeMo Framework is easy. State-of-the-art pretrained
NeMo models are freely available on [Hugging Face
Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia)
and [NVIDIA
NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
These models can be used to generate text or images, transcribe audio,
and synthesize speech in just a few lines of code.

We have extensive
[tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)
that can be run on [Google Colab](https://colab.research.google.com) or
with our [NGC NeMo Framework
Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
We also have
[playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)
for users who want to train NeMo models with the NeMo Framework
Launcher.

For advanced users who want to train NeMo models from scratch or
fine-tune existing NeMo models, we have a full suite of [example
scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) that support
multi-GPU/multi-node training.

## Key Features

- [Large Language Models](nemo/collections/nlp/README.md)
- [Multimodal](nemo/collections/multimodal/README.md)
- [Automatic Speech Recognition](nemo/collections/asr/README.md)
- [Text to Speech](nemo/collections/tts/README.md)
- [Computer Vision](nemo/collections/vision/README.md)

## Requirements

- Python 3.10 or above
- Pytorch 2.5 or above
- NVIDIA GPU (if you intend to do model training)

## Developer Documentation

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Install NeMo Framework

The NeMo Framework can be installed in a variety of ways, depending on
your needs. Depending on the domain, you may find one of the following
installation methods more suitable.

- [Conda / Pip](#conda--pip): Install NeMo-Framework with native Pip into a virtual environment.
  - Used to explore NeMo on any supported platform.
  - This is the recommended method for ASR and TTS domains.
  - Limited feature-completeness for other domains.
- [NGC PyTorch container](#ngc-pytorch-container): Install NeMo-Framework from source with feature-completeness into a highly optimized container.
  - For users that want to install from source in a highly optimized container.
- [NGC NeMo container](#ngc-nemo-container): Ready-to-go solution of NeMo-Framework
  - For users that seek highest performance.
  - Contains all dependencies installed and tested for performance and convergence.

### Support matrix

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

- Fully supported: Max performance and feature-completeness.
- Limited supported: Used to explore NeMo.
- No support yet: In development.
- Deprecated: Support has reached end of life.

Please refer to the following table for current support levels:

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

### Conda / Pip

Install NeMo in a fresh Conda environment:

```bash
conda create --name nemo python==3.10.12
conda activate nemo
```

#### Pick the right version

NeMo-Framework publishes pre-built wheels with each release.
To install nemo_toolkit from such a wheel, use the following installation method:

```bash
pip install "nemo_toolkit[all]"
```

If a more specific version is desired, we recommend a Pip-VCS install. From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit, branch, or tag that you would like to install.  
To install nemo_toolkit from this Git reference `$REF`, use the following installation method:

```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout @${REF:-'main'}
pip install '.[all]'
```

#### Install a specific Domain

To install a specific domain of NeMo, you must first install the
nemo_toolkit using the instructions listed above. Then, you run the
following domain-specific commands:

```bash
pip install nemo_toolkit['all'] # or pip install "nemo_toolkit['all']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['asr'] # or pip install "nemo_toolkit['asr']@git+https://github.com/NVIDIA/NeMo@$REF:-'main'}"
pip install nemo_toolkit['nlp'] # or pip install "nemo_toolkit['nlp']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['tts'] # or pip install "nemo_toolkit['tts']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['vision'] # or pip install "nemo_toolkit['vision']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['multimodal'] # or pip install "nemo_toolkit['multimodal']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
```

### NGC PyTorch container

**NOTE: The following steps are supported beginning with 24.04 (NeMo-Toolkit 2.3.0)**

We recommended that you start with a base NVIDIA PyTorch container:
nvcr.io/nvidia/pytorch:25.01-py3.

If starting with a base NVIDIA PyTorch container, you must first launch
the container:

```bash
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
```

From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit/branch/tag that you want to install.  
To install nemo_toolkit including all of its dependencies from this Git reference `$REF`, use the following installation method:

```bash
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout ${REF:-'main'}
bash docker/common/install_dep.sh --library all
pip install ".[all]"
```

## NGC NeMo container

NeMo containers are launched concurrently with NeMo version updates.
NeMo Framework now supports LLMs, MMs, ASR, and TTS in a single
consolidated Docker container. You can find additional information about
released containers on the [NeMo releases
page](https://github.com/NVIDIA/NeMo/releases).

To use a pre-built container, run the following code:

```bash
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}
```

## Future Work

The NeMo Framework Launcher does not currently support ASR and TTS
training, but it will soon.

## Discussions Board

FAQ can be found on the NeMo [Discussions
board](https://github.com/NVIDIA/NeMo/discussions). You are welcome to
ask questions or start discussions on the board.

## Contribute to NeMo

We welcome community contributions! Please refer to
[CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md)
for the process.

## Publications

We provide an ever-growing list of
[publications](https://nvidia.github.io/NeMo/publications/) that utilize
the NeMo Framework.

To contribute an article to the collection, please submit a pull request
to the `gh-pages-src` branch of this repository. For detailed
information, please consult the README located at the [gh-pages-src
branch](https://github.com/NVIDIA/NeMo/tree/gh-pages-src#readme).

## Blogs

<!-- markdownlint-disable -->
<details open>
  <summary><b>Large Language Models and Multimodal Models</b></summary>
    <details>
      <summary>
        <a href="https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/">
          Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso
        </a> (2024/03/06)
      </summary>
      Bria, a Tel Aviv startup at the forefront of visual generative AI for enterprises now leverages the NVIDIA NeMo Framework. 
      The Bria.ai platform uses reference implementations from the NeMo Multimodal collection, trained on NVIDIA Tensor Core GPUs, to enable high-throughput and low-latency image generation. 
      Bria has also adopted NVIDIA Picasso, a foundry for visual generative AI models, to run inference.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/">
          New NVIDIA NeMo Framework Features and NVIDIA H200
        </a> (2023/12/06)
      </summary>
      NVIDIA NeMo Framework now includes several optimizations and enhancements, 
      including: 
      1) Fully Sharded Data Parallelism (FSDP) to improve the efficiency of training large-scale AI models, 
      2) Mix of Experts (MoE)-based LLM architectures with expert parallelism for efficient LLM training at scale, 
      3) Reinforcement Learning from Human Feedback (RLHF) with TensorRT-LLM for inference stage acceleration, and 
      4) up to 4.2x speedups for Llama 2 pre-training on NVIDIA H200 Tensor Core GPUs.
      <br><br>
      <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility">
      <img src="https://github.com/sbhavani/TransformerEngine/blob/main/docs/examples/H200-NeMo-performance.png" alt="H200-NeMo-performance" style="width: 600px;"></a>
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://blogs.nvidia.com/blog/nemo-amazon-titan/">
          NVIDIA now powers training for Amazon Titan Foundation models
        </a> (2023/11/28)
      </summary>
      NVIDIA NeMo Framework now empowers the Amazon Titan foundation models (FM) with efficient training of large language models (LLMs). 
      The Titan FMs form the basis of Amazon’s generative AI service, Amazon Bedrock. 
      The NeMo Framework provides a versatile framework for building, customizing, and running LLMs.
      <br><br>
    </details>
</details>
<!-- markdownlint-enable -->

## Licenses

- [NeMo GitHub Apache 2.0
  license](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file#readme)
- NeMo is licensed under the [NVIDIA AI PRODUCT
  AGREEMENT](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/).
  By pulling and using the container, you accept the terms and
  conditions of this license.