## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import train
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule


def get_args():
    parser = argparse.ArgumentParser(description='Train a small GPT model using NeMo 2.0')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument('--experiment-dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--data-path', type=str, help="Path to data file")
    parser.add_argument('--vocab-path', type=str, help="Path to vocab file")
    parser.add_argument('--merges-path', type=str, help="Path to merges file")
    parser.add_argument('--index-mapping-dir', type=str, help="directory to write index mappings to")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    seq_length = 2048

    tokenizer = get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=args.vocab_path,
        merges_file=args.merges_path,
    )
    data = PreTrainingDataModule(
        paths=args.data_path,
        seq_length=2048,
        global_batch_size=32,
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
    )
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)
    strategy = nl.MegatronStrategy()
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=5000,
        enable_nemo_ckpt_io=False,
    )
    callbacks = [checkpoint_callback]

    loggers = []
    tensorboard_logger = TensorBoardLogger(
        save_dir='dummy',  ## NOTE: this gets overwritten by default
    )
    loggers.append(tensorboard_logger)

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        min_lr=6e-5,
        use_distributed_optimizer=False,
        bf16=True,
    )
    opt = MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=2,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=False),
    )

    nemo_logger = NeMoLogger(
        dir=args.experiment_dir,
    )

    train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )
