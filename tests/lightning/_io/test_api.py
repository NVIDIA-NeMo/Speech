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

import os
from functools import partial
from pathlib import Path

import pytest
import yaml
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
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
    def test_reload_ckpt(self, tmpdir, partial_function_with_pos_and_key_args):
        trainer = nl.Trainer(
            devices=1,
            accelerator="cpu",
            strategy=nl.MegatronStrategy(),
            logger=TensorBoardLogger("tb_logs", name="my_model"),
        )
        tokenizer = get_nmt_tokenizer("megatron", "GPT2BPETokenizer")
        model = llm.GPTModel(
            llm.GPTConfig(
                num_layers=2,
                hidden_size=1024,
                ffn_hidden_size=4096,
                num_attention_heads=8,
            ),
            tokenizer=tokenizer,
        )

        ckpt = io.TrainerContext(model, trainer, extra={"dummy": partial_function_with_pos_and_key_args})
        ckpt.io_dump(tmpdir, yaml_attrs=["model"])
        loaded = io.load_context(tmpdir)

        assert loaded.model.config.seq_length == ckpt.model.config.seq_length
        assert loaded.model.__io__.tokenizer.vocab_file.startswith(str(tmpdir))
        assert loaded.model.__io__.tokenizer.merges_file.startswith(str(tmpdir))

        loaded_func = loaded.extra["dummy"]
        assert loaded_func(b=2) == partial_function_with_pos_and_key_args(b=2)

        model_yaml = Path(tmpdir) / "model.yaml"
        assert model_yaml.exists()

        observed = yaml.safe_load(model_yaml.read_text())
        expected = yaml.safe_load((Path(ARTIFACTS_DIR) / "model.yaml").read_text())
        assert observed.keys() == expected.keys()
