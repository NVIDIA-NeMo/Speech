# NeMo LLM Recipes

This directory contains recipes for pre-training and fine-tuning large language models (LLMs) using NeMo.

A recipe in NeMo is a Python file that defines a complete configuration for training or fine-tuning an LLM. Each recipe typically includes:

1. Model configuration: Defines the architecture and hyperparameters of the LLM.
2. Training configuration: Specifies settings for the PyTorch Lightning Trainer, including distributed training strategies.
3. Data configuration: Sets up the data pipeline, including batch sizes and sequence lengths.
4. Optimization configuration: Defines the optimizer and learning rate schedule.
5. Logging and checkpointing configuration: Specifies how to save model checkpoints and log training progress.

Recipes are designed to be modular and extensible, allowing users to easily customize settings for their specific use cases.

## Usage

### Command Line Interface

You can use these recipes via the NeMo CLI (provided by [NeMo-Run](https://github.com/NVIDIA/NeMo-Run)):

```bash
nemo llm <task> --factory <recipe_name>
```
Where:
- `<task>` is either `pretrain` or `finetune`
- `<recipe_name>` is the name of the recipe (e.g. `llama3_8b`)

For example:
```bash
nemo llm pretrain --factory llama3_8b
```

> [!IMPORTANT]
> When launching the recipes with multiple processes (i.e. on multiple GPUs), add the `-y` option to the command to avoid user confirmation prompts.
> For example, `nemo llm pretrain --factory llama3_8b -y`

### Customizing Parameters

You can override any parameter in the recipe:

```bash
nemo llm pretrain --factory llama3_8b trainer.max_steps=2000
```

For more details around running recipes, see [pre-train](../../../../examples/llm/pretrain/README.md).

## Adding a New Recipe

See [ADD-RECIPE.md](ADD-RECIPE.md) for instructions on how to add a new recipe.