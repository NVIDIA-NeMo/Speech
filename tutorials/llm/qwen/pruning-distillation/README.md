# Qwen3-8B Pruning and Distillation with NeMo 2.0 Framework

[NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) is the library (referred to as **Model Optimizer**, or **ModelOpt**) comprising state-of-the-art model optimization techniques including quantization, distillation, pruning, and speculative decoding to compress models. We will use this library to perform the pruning and distillation on [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) in [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)

[LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796) provides details pruning and distillation on Llama 3.1 as described in the [tech report](https://arxiv.org/abs/2408.11796).

[How to Prune and Distill Llama-3.1 8B to an NVIDIA Llama-3.1-Minitron 4B Model](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/) provides practical and effective structured compression best practices for LLMs that combine depth, width, attention, and MLP pruning with knowledge distillation-based retraining.

[Supercharge Edge AI With High‑Accuracy Reasoning Using NVIDIA Nemotron Nano 2 9B](https://huggingface.co/blog/nvidia/supercharge-ai-reasoning-with-nemotron-nano-2) talks about how state-of-the-art reasoning model [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) was created by pruning and distilling a 12B Hybrid Mamba Transformer model which is also supported by TensorRT Model Optimizer.

## Objectives

This tutorial demonstrates how to perform depth-pruning, width-pruning, and distillation on **Qwen3-8B** using the [WikiText](https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1) dataset with the NeMo Framework. We will start with a HuggingFace checkpoint and convert it to NeMo format to use for pruning and distillation and later convert the distilled model back to HuggingFace format. The `WikiText` language modeling dataset comprises over 100 million tokens extracted from verified Good and Featured articles on Wikipedia. While this is the most easy to get started with, in practice, we recommend using bigger, more recent and much higher quality datasets like [ClimbMix](https://huggingface.co/datasets/OptimalScale/ClimbMix) or [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1).

There are two methods to prune a model: depth-pruning and width-pruning. We will explore both techniques, yielding 2 pruned models. These models will serve as starting points for distillation to create the final distilled models.

**NOTE:** Checkout the full list of supported models and prunable dimensions [here](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/pruning).

## Requirements

### System Configuration
- Access to at least 8 NVIDIA GPUs, each with a memory of at least 80GB (e.g., 8 x H100-80GB or 8 x A100-80GB).
- A Docker-enabled environment, with [NVIDIA Container Runtime](https://developer.nvidia.com/container-runtime) installed, which will make the container GPU-aware.

Get your Hugging Face [access token](https://huggingface.co/docs/hub/en/security-tokens), which will be used to download gated models or datasets.

**NOTE:** The default configuration in the notebook runs on 8 x 80GB NVIDIA GPUs. However, you can potentially reduce the Tensor Parallel size (`TENSOR_PARALLEL_SIZE`) along with the Micro-Batchsize (`MICRO_BATCH_SIZE`) in the distillation scripts to accommodate lower resource availability.

## Create a Pruned and Distilled Model with NeMo Framework

For pruning and distilling the model, you will use the NeMo Framework, which is available as a [Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo). These notebooks has been tested on `nvcr.io/nvidia/nemo:25.09` container with `nvidia-modelopt==0.35.1`.

1. Run the container using the following command. You will mount your local directory to `/workspace` so the model and dataset will be stored in a persistent location. If you are using your own model and dataset, you can change the paths in the notebooks accordingly.

```bash
export FW_VERSION=25.09
```

```bash
docker run \
  --gpus all \
  --shm-size=16GB \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v ${PWD}:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:$FW_VERSION bash
```

2. From within the container, copy the notebooks to your local directory so changes remain persistent (only if running first time).

```bash
cp -r /opt/NeMo/tutorials/llm/qwen/pruning-distillation/* /workspace
```

3. From within the container, login with your Hugging Face token to download any gated models or datasets (not required if you have already downloaded the model and datasets).

```bash
huggingface-cli login --token <YOUR_HF_ACCESS_TOKEN>
```

4. Start the Jupyter lab:

```bash
pip install --upgrade ipywidgets notebook
jupyter lab --ip 0.0.0.0 --port=8888 --allow-root
```

5. Then, navigate to this directory which contains a list of notebooks that cover all the steps to create a distilled 6B model from Qwen3-8B.

This workflow is structured into four notebooks:
  1. [Prepare the model and dataset](./01_model_and_data_preparation.ipynb) to convert HuggingFace model to NeMo format and tokenize the dataset
  2. [Prune the model](./02_pruning.ipynb) to create a pruned model via either depth-pruning or width-pruning
  3. [Distill knowledge from teacher into pruned student](./03_distillation.ipynb)
  4. [Compare depth and width pruned models](./04_depth_vs_width_pruning_comparison.ipynb)

> `NOTE:` We are exploring two methods to prune the model: depth-pruning and width-pruning. Per the [tech report](https://arxiv.org/pdf/2408.11796), we can observe that width-pruning generally outperforms depth-pruning while depth pruned model is generally faster at same number of parameters so users can choose to perform either depth-pruning or width-pruning or both methods simultaneously.
