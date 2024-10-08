import nemo_run as run
import pytest

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.nemotron import Nemotron3Config8B, NemotronModel
from nemo.collections.llm.recipes import nemotron3_8b
from nemo.lightning import AutoResume, Trainer


class TestNemotron3_8B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return nemotron3_8b

    def test_model(self, recipe_module):
        model = recipe_module.model()
        assert isinstance(model, run.Config)
        assert model.__fn_or_cls__ == NemotronModel

    def test_model_config_parameters(self, recipe_module):
        model = recipe_module.model()
        nemotron_config = model.config
        assert isinstance(nemotron_config, run.Config)
        assert nemotron_config.__fn_or_cls__ == Nemotron3Config8B
        assert nemotron_config.num_layers == 32
        assert nemotron_config.hidden_size == 4096
        assert nemotron_config.seq_length == 4096
        assert nemotron_config.num_attention_heads == 32

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == NemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 32
        assert recipe.data.micro_batch_size == 2

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == NemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == SquadDataModule
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 32
        assert recipe.data.micro_batch_size == 2
