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

import lightning.pytorch as pl
import pytest
import torch
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm


class DummyTokenizer:
    def __init__(self):
        self.vocab_size = 30000


class TestMegatronMixedPrecision:
    """Unit tests for the MegatronMixedPrecision class."""

    @pytest.mark.run_only_on('GPU')
    def test_precision_plugin_fp8_passed(self):
        """Test __init__ with default parameters."""

        class TrainerHook(nl.Trainer):
            def connect(self, model: pl.LightningModule) -> None:
                assert model.config.bf16 == False
                assert model.config.fp8 is None
                super().connect(model)
                assert model.config.fp8 == 'e4m3'
                assert model.config.bf16 == True

        trainer = TrainerHook(
            devices=2,
            accelerator="gpu",
            max_steps=2,
            strategy=nl.MegatronStrategy(
                tensor_model_parallel_size=2,
                sequence_parallel=True,
                ckpt_load_optimizer=False,
                ckpt_save_optimizer=False,
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", fp8='e4m3'),
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
        )

        optim = nl.MegatronOptimizerModule(
            config=OptimizerConfig(
                optimizer="adam",
                lr=1e-5,
                use_distributed_optimizer=False,
                fp16=True,
                params_dtype=torch.float32,
            ),
        )
        config = llm.Llama2Config7B()
        config.num_layers = 2
        model = llm.LlamaModel(config, tokenizer=DummyTokenizer(), optim=optim)
        trainer.strategy.connect(model)

    @pytest.mark.run_only_on('GPU')
    def test_precision_plugin_precision_params_override(self):
        """Test __init__ with default parameters."""
        trainer = nl.Trainer(
            devices=2,
            accelerator="gpu",
            max_steps=2,
            strategy=nl.MegatronStrategy(
                tensor_model_parallel_size=2,
                sequence_parallel=True,
                ckpt_load_optimizer=False,
                ckpt_save_optimizer=False,
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
        )

        optim = nl.MegatronOptimizerModule(
            config=OptimizerConfig(
                optimizer="adam",
                lr=1e-5,
                use_distributed_optimizer=False,
                fp16=True,
                params_dtype=torch.float32,
            ),
        )
        config = llm.Llama2Config7B()
        config.num_layers = 2
        config.fp16 = True
        config.bf16 = False
        model = llm.LlamaModel(config, tokenizer=DummyTokenizer(), optim=optim)
        trainer.strategy.connect(model)
        assert optim.config.bf16 is not None
        assert optim.config.fp16 is not None
        assert optim.config.bf16 == True
        assert optim.config.fp16 == False
        assert model.config.fp16 == False
        assert model.config.bf16 == True
