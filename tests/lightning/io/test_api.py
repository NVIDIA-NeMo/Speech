import transformer_engine as te
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import io


class TestLoad:
    def test_reload_ckpt(self, tmpdir):
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

        ckpt = io.TrainerContext(model, trainer)
        ckpt.io_dump(tmpdir)
        loaded = io.load_context(tmpdir)

        assert loaded.model.config.seq_length == ckpt.model.config.seq_length
        assert loaded.model.__io__.tokenizer.vocab_file.startswith(str(tmpdir))
        assert loaded.model.__io__.tokenizer.merges_file.startswith(str(tmpdir))
