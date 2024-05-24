from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import io


class TestLoad:
    def test_reload_ckpt(self, tmpdir):
        trainer = nl.Trainer(devices=1, accelerator="cpu", strategy=nl.MegatronStrategy())
        # model = llm.Mistral7BModel()
        model = llm.GPTModel(
            llm.GPTConfig(
                num_layers=2,
                hidden_size=1024,
                ffn_hidden_size=4096,
                num_attention_heads=8,
            )
        )

        ckpt = io.TrainerCheckpoint(model, trainer)
        ckpt.io_dump(tmpdir)
        loaded = io.load_ckpt(tmpdir)

        assert loaded.model.config.seq_length == ckpt.model.config.seq_length
