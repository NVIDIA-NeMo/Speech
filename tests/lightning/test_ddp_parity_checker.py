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

import argparse
import os

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.lightning.pytorch.callbacks import DdpParityChecker


def make_parser():
    parser = argparse.ArgumentParser(description='Train a small GPT model using NeMo 2.0')
    parser.add_argument('--data-path', type=str, help="Path to data file")
    parser.add_argument('--vocab-path', type=str, help="Path to vocab file")
    parser.add_argument('--merges-path', type=str, help="Path to merges file")

    return parser


def wrap_config(config, trainer):
    class ConfigWrapper(type(config)):
        def configure_model(self, tokenizer, vp_stage=None) -> "MCoreGPTModel":
            return make_byzantine_model_wrapper(super().configure_model(tokenizer), trainer)

    config.__class__ = ConfigWrapper
    return config


def make_byzantine_model_wrapper(model, trainer):
    class ByzantineModel(type(model)):
        def forward(self, *ans, **kwargs):
            ans = super().forward(*ans, **kwargs)
            with torch.no_grad():
                import random

                rank = int(os.environ['LOCAL_RANK'])
                if rank != 1:
                    return ans
                for opt in trainer.strategy.model.optim._optimizers:
                    for g in opt.param_groups:
                        for param in g['params']:
                            param.fill_(random.uniform(0, 1))
            return ans

    model.__class__ = ByzantineModel
    return model


@pytest.mark.skip(reason="tested with GH")
def test_failing(trainer, ddp_parity, optim, data, tokenizer):
    config = llm.Llama2Config7B(num_layers=2)
    config = wrap_config(config, trainer)
    model = llm.LlamaModel(config, tokenizer=tokenizer, optim=optim)
    trainer.fit(model, data)


@pytest.mark.skip(reason="tested with GH")
def test_working(trainer, ddp_parity, optim, data, tokenizer):
    config = llm.Llama2Config7B(num_layers=2)
    model = llm.LlamaModel(config, tokenizer=tokenizer, optim=optim)
    trainer.fit(model, data)


def make_trainer_optim(args):
    ddp_parity = DdpParityChecker(1)
    trainer = nl.Trainer(
        devices=2,
        max_steps=4,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
        ),
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        limit_val_batches=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        logger=None,
        callbacks=[ddp_parity],
    )

    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=1e-5,
            use_distributed_optimizer=False,
            fp16=False,
            bf16=True,
            params_dtype=torch.float32,
        ),
    )

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

    return trainer, ddp_parity, optim, data, tokenizer


@pytest.mark.skip(reason="tested with GH")
def main():
    args = make_parser().parse_args()
    trainer, ddp_parity, optim, data, tokenizer = make_trainer_optim(args)
    test_failing(trainer, ddp_parity, optim, data, tokenizer)
    if trainer.should_stop != True:
        raise ValueError("DDP parity checking failed.")

    try:
        test_working(*make_trainer_optim(args))
        print("DDP parity checking worked as expected")
    except:
        raise


if __name__ == "__main__":
    main()
