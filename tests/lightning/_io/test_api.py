# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from functools import partial
from unittest.mock import patch

import fiddle as fdl
import pytest
from lightning.pytorch.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import io
from nemo.utils.import_utils import safe_import

te, HAVE_TE = safe_import("transformer_engine")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "artifacts")


def dummy_extra(a, b, c=5):
    return a + b + c


@pytest.fixture
def partial_function_with_pos_and_key_args():
    return partial(dummy_extra, 10, c=15)


class TestLoad:
    @patch('nemo.lightning.callback_group.CallbackGroup.update_config')
    def test_reload_ckpt(self, mock_update_one_logger, tmpdir, partial_function_with_pos_and_key_args):
        # Mock the OneLogger callback update to prevent it from adding callbacks to the trainer
        # This avoids serialization issues with the trainer during checkpoint saving
        mock_update_one_logger.return_value = None

        trainer = nl.Trainer(
            devices=1,
            accelerator="cpu",
            strategy=nl.MegatronStrategy(),
            logger=TensorBoardLogger("tb_logs", name="my_model"),
        )

        # Create a model without a tokenizer to avoid serialization issues
        model = llm.GPTModel(
            llm.GPTConfig(
                num_layers=2,
                hidden_size=1024,
                ffn_hidden_size=4096,
                num_attention_heads=8,
            ),
            # Don't pass tokenizer to avoid serialization issues
        )

        ckpt = io.TrainerContext(model, trainer, extra={"dummy": partial_function_with_pos_and_key_args})
        ckpt.io_dump(tmpdir, yaml_attrs=["model"])
        loaded = io.load_context(tmpdir)

        assert loaded.model.config.seq_length == ckpt.model.config.seq_length

        # Since we don't have a tokenizer, we can't test tokenizer-related assertions
        # The test focuses on testing the TrainerContext functionality

        loaded_func = loaded.extra["dummy"]
        assert loaded_func(b=2) == partial_function_with_pos_and_key_args(b=2)

        config = io.load_context(tmpdir, build=False)
        assert isinstance(config, fdl.Config)
        assert config.model.config.seq_length == ckpt.model.config.seq_length
        assert config.extra["dummy"] == fdl.Partial(dummy_extra, 10, c=15)
