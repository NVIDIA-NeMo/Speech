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
from nemo.lightning import AutoResume, NeMoLogger
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from tests.lightning.mcore_microbatch_utils import reconfigure_num_microbatches_calculator_manager

VOCAB_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json"
MERGES_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt"
DATA_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document"
EXP_DIR = '/tmp/nemo_exp/'


def teardown(exp_dir=EXP_DIR):
    import shutil

    shutil.rmtree(exp_dir)


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
            assert torch.all(p == 0), "Expected params (scratch) to be zero"
        with torch.no_grad():
            for p in pl_module.parameters():
                p.fill_(random.uniform(0, 1))


class ValidateModelRestoration(Callback):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for p in pl_module.parameters():
            p.detach().zero_()
        self.called_on_load_checkpoint = False

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        self.called_on_load_checkpoint = True

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for p in pl_module.parameters():
            assert not torch.all(p == 0), "Expected params (resume) to be non-zero"
        assert hasattr(self, 'called_on_load_checkpoint')
        assert self.called_on_load_checkpoint == True, "Expected to have called on_load_checkpoint"


def setup_data(mbs=1, gbs=2, seq_length=2048):
    tokenizer = get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=VOCAB_PATH,
        merges_file=MERGES_PATH,
    )

    data = PreTrainingDataModule(
        paths=DATA_PATH,
        seq_length=2048,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        seed=1234,
        tokenizer=tokenizer,
    )
    return data


def make_model_optim(tokenizer, mbs=1, gbs=2, seq_length=2048):
    gpt_config = llm.GPTConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=12,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=False,
    )
    model = llm.GPTModel(gpt_config, tokenizer=tokenizer)

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

    return model, opt


def run_train_from_scratch(mbs, gbs, num_dev):
    data = setup_data(mbs, gbs)
    model, opt = make_model_optim(data.tokenizer, mbs, gbs)
    # Other tests might have different configs, so need to configure explicitly.
    with reconfigure_num_microbatches_calculator_manager(
        0,
        None,
        gbs,
        mbs,
        data_parallel_size=num_dev,
    ):
        trainer = nl.Trainer(
            devices=num_dev,
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
                version='v1',
                use_datetime_version=True,
                update_logger_directory=True,
                wandb=None,
            ),
            resume=AutoResume(
                resume_if_exists=True,
                resume_ignore_no_checkpoint=True,
            ),
            tokenizer='data',
            optim=opt,
        )
        trainer._teardown()


def run_resume_train(mbs, gbs, num_dev):
    data = setup_data(mbs, gbs)
    model, opt = make_model_optim(data.tokenizer, mbs, gbs)
    # Other tests might have different configs, so need to configure explicitly.
    with reconfigure_num_microbatches_calculator_manager(
        0,
        None,
        gbs,
        mbs,
        data_parallel_size=num_dev,
    ):
        trainer = nl.Trainer(
            devices=num_dev,
            max_steps=1,
            accelerator="gpu",
            strategy=nl.MegatronStrategy(),
            callbacks=[ValidateOptStateRestoration(), ValidateModelRestoration()],
            log_every_n_steps=1,
            limit_val_batches=2,
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        )
        from nemo.lightning.pytorch.strategies.utils import RestoreConfig

        train(
            model=model,
            data=data,
            trainer=trainer,
            tokenizer='data',
            optim=opt,
            log=NeMoLogger(
                log_dir=EXP_DIR,
                version='v1',
                use_datetime_version=True,
                update_logger_directory=True,
                wandb=None,
            ),
            resume=AutoResume(
                resume_if_exists=True,
                resume_ignore_no_checkpoint=False,
                resume_from_path=f'{EXP_DIR}default/v1/checkpoints/default--None=0.0000-epoch=0/',
            ),
        )
        trainer._teardown()


@pytest.mark.run_only_on('GPU')
def test_optim_state_restoration():
    mbs, gbs = 1, 2
    num_devices = 1
    try:
        run_train_from_scratch(mbs, gbs, num_devices)
        run_resume_train(mbs, gbs, num_devices)
    finally:
        teardown()
