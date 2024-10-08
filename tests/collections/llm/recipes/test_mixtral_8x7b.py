import nemo_run as run
import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x7B, MixtralModel
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import mixtral_8x7b
from nemo.lightning import AutoResume, Trainer


class TestMixtral8x7B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return mixtral_8x7b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == MixtralModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == MixtralConfig8x7B

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 8

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 8
        assert trainer_config.strategy.context_parallel_size == 1
        assert trainer_config.strategy.sequence_parallel is False
        assert trainer_config.strategy.expert_model_parallel_size == 8

        # Check DDP configuration
        assert isinstance(trainer_config.strategy.ddp, run.Config)
        assert trainer_config.strategy.ddp.__fn_or_cls__ == DistributedDataParallelConfig
        assert trainer_config.strategy.ddp.check_for_nan_in_grad is True
        assert trainer_config.strategy.ddp.grad_reduce_in_fp32 is True

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
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 512
        assert recipe.data.micro_batch_size == 1

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == MixtralModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == SquadDataModule
        assert recipe.data.seq_length == 2048
        assert recipe.data.global_batch_size == 128
        assert recipe.data.micro_batch_size == 1
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.peft.target_modules == ['linear_qkv', 'linear_proj']
        assert recipe.peft.dim == 32

    def test_trainer_parallelism_options(self, recipe_module):
        trainer_config = recipe_module.trainer(
            tensor_parallelism=4,
            pipeline_parallelism=4,
            context_parallelism=2,
            sequence_parallelism=False,
            expert_parallelism=2,
        )
        assert trainer_config.strategy.tensor_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_model_parallel_size == 4
        assert trainer_config.strategy.context_parallel_size == 2
        assert trainer_config.strategy.sequence_parallel is False
        assert trainer_config.strategy.expert_model_parallel_size == 2

    def test_model_config_parameters(self, recipe_module):
        model_config = recipe_module.model()
        mixtral_config = model_config.config
        assert mixtral_config.num_layers == 32
        assert mixtral_config.hidden_size == 4096
        assert mixtral_config.num_attention_heads == 32
        assert mixtral_config.seq_length == 4096
        assert mixtral_config.num_moe_experts == 8
