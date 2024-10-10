## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import finetune
from nemo.collections.llm.t5.data import SquadDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule


def get_args():
    parser = argparse.ArgumentParser(description='Train a small T5 model using NeMo 2.0')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument('--experiment-dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--experiment-name', type=str, help="name of experiment")
    parser.add_argument('--wandb-project', type=str, default=None, help="wandb project name")
    parser.add_argument('--checkpoint-path', type=str, help="Path to checkpoint dir")
    parser.add_argument('--index-mapping-dir', type=str, help="directory to write index mappings to")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    tokenizer = get_nmt_tokenizer(
        "megatron",
        "BertWordPieceCase",
    )

    data = SquadDataModule(
        seq_length=512,
        seq_length_dec=128,
        micro_batch_size=16,
        global_batch_size=128,
        tokenizer=tokenizer,
        num_workers=4,
    )

    t5_config = llm.t5.model.t5.T5Config(
        num_layers=12,
        encoder_num_layers=12,
        hidden_size=768,
        ffn_hidden_size=3072,
        num_attention_heads=12,
        kv_channels=64,
        init_method_std=0.015,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        max_position_embeddings=512,
    )
    model = llm.t5.model.t5.T5Model(t5_config, tokenizer=data.tokenizer)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.float32,
        ckpt_load_optimizer=False,
        # ckpt_load_optimizer=True,
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=5000,
    )
    callbacks = [checkpoint_callback]

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_path=args.checkpoint_path,
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=2.0e-5,
        use_distributed_optimizer=False,
        bf16=False,
        weight_decay=0.1,
    )
    opt = MegatronOptimizerModule(
        config=opt_config,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=50,
        plugins=nl.MegatronMixedPrecision(precision="32"),
    )

    if args.wandb_project is not None:
        wandb_logger = WandbLogger(
            name=args.experiment_name,
            project=args.wandb_project,
            log_model="all",
        )
    else:
        wandb_logger = None
    nemo_logger = NeMoLogger(
        name=args.experiment_name,
        use_datetime_version=False,
        log_dir=args.experiment_dir,
        wandb=wandb_logger,
    )

    finetune(
        model=model,
        resume=resume,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
    )
