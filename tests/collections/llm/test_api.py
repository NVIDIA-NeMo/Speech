# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile

import nemo_run as run
import pytest
import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import _validate_config
from nemo.collections.llm.gpt.model.llama import Llama3Config8B, LlamaModel


class TestValidateConfig:

    def reset_configs(self):
        model = LlamaModel(config=run.Config(Llama3Config8B))
        data = llm.MockDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
        trainer = nl.Trainer(strategy=nl.MegatronStrategy())
        return model, data, trainer

    def test_model_validation(self):
        model, data, trainer = self.reset_configs()
        _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.seq_length = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.num_layers = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.hidden_size = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.num_attention_heads = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.ffn_hidden_size = 0
            _validate_config(model, data, trainer)

    def test_data_validation(self):
        model, data, trainer = self.reset_configs()
        _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.micro_batch_size = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.global_batch_size = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.seq_length = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.micro_batch_size = 3
            data.global_batch_size = 128
            _validate_config(model, data, trainer)

    def test_trainer_validatiopn(self):
        model, data, trainer = self.reset_configs()
        _validate_config(model, data, trainer)

        # Basic validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.tensor_model_parallel_size = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.pipeline_model_parallel_size = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.context_parallel_size = 0
            _validate_config(model, data, trainer)

        # DP validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.tensor_model_parallel_size = 8
            trainer.strategy.pipeline_model_parallel_size = 2
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.tensor_model_parallel_size = 3
            trainer.strategy.pipeline_model_parallel_size = 2
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.global_batch_size = 3
            data.micro_batch_size = 1
            trainer.strategy.tensor_model_parallel_size = 2
            trainer.strategy.pipeline_model_parallel_size = 2
            _validate_config(model, data, trainer)

        # TP/SP validation
        model, data, trainer = self.reset_configs()
        trainer.strategy.tensor_model_parallel_size = 1
        trainer.strategy.sequence_parallel = True
        _validate_config(model, data, trainer)
        assert trainer.strategy.sequence_parallel == False

        # PP/VP validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.pipeline_model_parallel_size = 2
            trainer.strategy.pipeline_dtype = None
            _validate_config(model, data, trainer)

        model, data, trainer = self.reset_configs()
        trainer.strategy.pipeline_model_parallel_size = 1
        trainer.strategy.virtual_pipeline_model_parallel_size = 2
        trainer.strategy.pipeline_dtype = torch.bfloat16
        _validate_config(model, data, trainer)
        assert trainer.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer.strategy.pipeline_dtype is None

        # CP validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.seq_length = 5
            trainer.strategy.context_parallel_size = 2
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.seq_length = 2
            trainer.strategy.context_parallel_size = 2
            _validate_config(model, data, trainer)

        # EP validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.num_moe_experts = None
            trainer.strategy.expert_model_parallel_size = 2
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.num_moe_experts = 3
            trainer.strategy.expert_model_parallel_size = 2
            _validate_config(model, data, trainer)


class TestImportCkpt:

    def test_output_path_exists_no_overwrite(self):
        """Test that an error is raised when the output path exists and overwrite is set to False."""

        with pytest.raises(FileExistsError), tempfile.TemporaryDirectory() as output_path:
            llm.import_ckpt(
                model=llm.LlamaModel(config=llm.Llama32Config1B()),
                source="hf://meta-llama/Llama-3.2-1B",
                output_path=output_path,
                overwrite=False,
            )


class TestExportCkpt:

    def test_output_path_exists_no_overwrite(self):
        """Test that an error is raised when the output path exists and overwrite is set to False."""

        with (
            pytest.raises(FileExistsError),
            tempfile.TemporaryDirectory() as output_path,
            tempfile.TemporaryDirectory() as path,
        ):
            llm.export_ckpt(
                path=path,
                target="hf",
                output_path=output_path,
                overwrite=False,
            )
