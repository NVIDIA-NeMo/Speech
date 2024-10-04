import nemo_run as run
import pytest
import torch

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config8B, LlamaModel
from nemo.collections.llm.recipes import llama3_8b_64k
from nemo.lightning import Trainer


class TestLlama3_8B_64k:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llama3_8b_64k

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == LlamaModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llama3Config8B
        assert model_config.config.seq_length == 65536

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 1

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 5
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 65536
        assert recipe.data.global_batch_size == 512
        assert recipe.data.micro_batch_size == 1

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_trainer_parallelism_options(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 5
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True

    def test_model_config_parameters(self, recipe_module):
        model_config = recipe_module.model()
        llama_config = model_config.config
        assert llama_config.num_layers == 32
        assert llama_config.hidden_size == 4096
        assert llama_config.num_attention_heads == 32
        assert llama_config.seq_length == 65536
