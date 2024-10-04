import nemo_run as run
import pytest
import torch

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x7B, MixtralModel
from nemo.collections.llm.recipes import mixtral_8x7b_16k
from nemo.lightning import Trainer
from nemo.utils.exp_manager import TimingCallback


class TestMixtral8x7B_16k:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return mixtral_8x7b_16k

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == MixtralModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == MixtralConfig8x7B
        assert model_config.config.seq_length == 16384
        assert model_config.config.max_position_embeddings == 16384

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 2

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 8
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True
        assert trainer_config.strategy.expert_model_parallel_size == 1

        # Check for TimingCallback
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == TimingCallback for cb in trainer_config.callbacks
        )

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == MixtralModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 16384
        assert recipe.data.global_batch_size == 512
        assert recipe.data.micro_batch_size == 1

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(2, 8), (4, 4), (8, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_trainer_parallelism_options(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 8
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True
        assert trainer_config.strategy.expert_model_parallel_size == 1

    def test_model_config_parameters(self, recipe_module):
        model_config = recipe_module.model()
        mixtral_config = model_config.config
        assert mixtral_config.num_layers == 32
        assert mixtral_config.hidden_size == 4096
        assert mixtral_config.num_attention_heads == 32
        assert mixtral_config.seq_length == 16384
        assert mixtral_config.max_position_embeddings == 16384
        assert mixtral_config.num_moe_experts == 8
