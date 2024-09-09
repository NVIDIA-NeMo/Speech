# How to Add a New Recipe

This guide explains the process of adding a new recipe to the NeMo LLM collection.

## Step 1: Create a New Python File

Create a new Python file in the `nemo/collections/llm/recipes/` directory. Name it according to the model and its specific configuration, e.g., `my_new_model_12b.py`.

## Step 2: Define the Model Configuration

Create a function called `model` to define the model configuration:

```python
NAME = "my_new_model_12b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    return run.Config(YourModel, config=run.Config(YourModelConfig))
```

## Step 3: Define the Trainer Configuration

Create a function called `trainer` to set up the trainer:

```python
def trainer(
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    # Add other parameters as needed
) -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        # Define your parallelism strategy here
    )
    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        # Add other trainer configurations
    )
    return trainer
```

## Step 4: Define the Recipe Configuration

Create a function called `pretrain_recipe` or `finetune_recipe` to define the recipe configuration:

```python
from nemo.collections.llm import pretrain

@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    # Add other parameters as needed
) -> run.Config[nl.PretrainRecipe]:
    return run.Config(
        nl.PretrainRecipe,
        model=model(),
        trainer=trainer(),
        # Add other recipe configurations
        data=run.Config(MockDataModule, seq_length=4096, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )
```

```python
from nemo.collections.llm import finetune

@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    # Add other parameters as needed
) -> run.Config[nl.FinetuneRecipe]:
    return run.Config(
        nl.FinetuneRecipe,
        model=model(),
        trainer=trainer(),
        # Add other recipe configurations
        data=run.Config(MockDataModule, seq_length=4096, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )
```


## Step 5: Import the recipe in the __init__.py file

Import the recipe in the [__init__.py](__init__.py) file in the same directory:

```python
from .my_new_model_12b import pretrain_recipe, finetune_recipe
```


## Step 6: Add tests for the recipe

Add tests for the recipe in the [tests](../../../../tests/collections/llm/recipes) directory. You can use [test_llama3_8b.py](../../../../tests/collections/llm/recipes/test_llama3_8b.py) as an example.
