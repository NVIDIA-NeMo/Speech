from functools import partial

import pytest
import transformer_engine as te
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import io


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
        ckpt.io_dump(tmpdir)
        loaded = io.load_context(tmpdir)

        assert loaded.model.config.seq_length == ckpt.model.config.seq_length
        assert loaded.model.__io__.tokenizer.vocab_file.startswith(str(tmpdir))
        assert loaded.model.__io__.tokenizer.merges_file.startswith(str(tmpdir))

        loaded_func = loaded.extra["dummy"]
        assert loaded_func(b=2) == partial_function_with_pos_and_key_args(b=2)
