import math
import random

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.callbacks import Callback

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import train
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule

VOCAB_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json"
MERGES_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt"
DATA_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document"
EXP_DIR = '/tmp/nemo_exp/'


class ValidateOptStateRestoration(Callback):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # PTL has no on_load_checkpoint_start event to be triggered before
        # the checkpoint restoration.
        opt_state = trainer.optimizers[0].state
        assert isinstance(opt_state, dict), "Expected state to be a dictionary"
        assert len(opt_state) == 0, "Expected state to be empty"

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        # This runs after the checkpoint restoration
        # on_load_checkpoint == on_load_checkpoint_end
        opt_state = trainer.optimizers[0].state
        assert isinstance(opt_state, dict), "Expected state to be a dictionary"
        assert len(opt_state) > 0, "Expected a non-empty state"
        for key, val in opt_state.items():
            for param in val.values():
                assert not torch.all(param == 0).item() and not torch.all(param == 1.0).item()


class ValidateOptStateScratchInit(Callback):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        opt_state = trainer.optimizers[0].state
        assert isinstance(opt_state, dict), "Expected state to be a dictionary"
        assert len(opt_state) == 0, "Expected state to be empty"

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        opt_state = trainer.optimizers[0].state
        assert isinstance(opt_state, dict), "Expected state to be a dictionary"
        assert len(opt_state) == 0, "Expected state to be empty"


class ValidateModelScratchInit(Callback):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for p in pl_module.parameters():
            p.detach().zero_()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for p in pl_module.parameters():
            assert torch.all(p == 0), "Expected params to be zero"
        with torch.no_grad():
            for p in pl_module.parameters():
                p.fill_(random.uniform(0, 1))


class ValidateModelRestoration(Callback):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for p in pl_module.parameters():
            p.detach().zero_()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for p in pl_module.parameters():
            assert not torch.all(p == 0), "Expected params to be non-zero"


def make_model_optim_data():
    seq_length = 2048
    tokenizer = get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=VOCAB_PATH,
        merges_file=MERGES_PATH,
    )

    data = PreTrainingDataModule(
        paths=DATA_PATH,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=2,
        seed=1234,
        tokenizer=tokenizer,
    )

    gpt_config = llm.GPTConfig(
        num_layers=12,
        hidden_size=768,
        ffn_hidden_size=3072,
        num_attention_heads=12,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=False,
    )
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)

    opt = MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer='adam',
            lr=1e-2,
            bf16=True,
            use_distributed_optimizer=False,
        ),
        lr_scheduler=CosineAnnealingScheduler(
            max_steps=50,
            min_lr=6e-5,
            warmup_steps=int(math.ceil(50 * 1 / 5)),
            interval="step",
            monitor="reduced_train_loss",
            constant_steps=int(math.ceil(50 * 1 / 5)),
        ),
    )

    return model, opt, data


def run_train_from_scratch():
    model, opt, data = make_model_optim_data()
    trainer = nl.Trainer(
        devices=2,
        max_steps=10,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(),
        callbacks=[ValidateOptStateScratchInit(), ValidateModelScratchInit()],
        log_every_n_steps=1,
        limit_val_batches=2,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    train(
        model=model,
        data=data,
        trainer=trainer,
        log=NeMoLogger(
            log_dir=EXP_DIR,
        ),
        tokenizer='data',
        optim=opt,
    )


def run_resume_train():
    model, opt, data = make_model_optim_data()
    trainer = nl.Trainer(
        devices=2,
        max_steps=1,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(),
        callbacks=[ValidateOptStateRestoration(), ValidateModelRestoration()],
        log_every_n_steps=1,
        limit_val_batches=2,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    train(
        model=model,
        data=data,
        trainer=trainer,
        log=NeMoLogger(
            log_dir=EXP_DIR,
        ),
        tokenizer='data',
        optim=opt,
        resume=nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        ),
    )


@pytest.mark.run_only_on('GPU')
def test_optim_state_restoration():
    run_train_from_scratch()
    run_resume_train()
