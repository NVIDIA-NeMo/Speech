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

import argparse

import lightning.pytorch as pl
import torch
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.io.mixin import track_io


def get_args():
    parser = argparse.ArgumentParser(description='Finetune a small GPT model using NeMo 2.0')
    parser.add_argument('--model', type=str.lower, choices=['mistral', 'mixtral'], help="model")
    parser.add_argument('--max-steps', type=int, default=9, help="number of devices")
    parser.add_argument('--mbs', type=int, default=2, help="micro batch size")
    parser.add_argument('--gbs', type=int, default=4, help="global batch size")
    parser.add_argument('--tp', type=int, default=1, help="tensor parallel size")
    parser.add_argument('--ep', type=int, default=1, help="expert parallel size")
    parser.add_argument('--dist-opt', action='store_true', help='use dist opt')
    parser.add_argument('--use-exclude', action='store_true', help='will use exclude_modules')
    return parser.parse_args()


def trainer(devices, tp, ep, sp, max_steps) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        sequence_parallel=sp,
    )

    return nl.Trainer(
        devices=max(ep, tp),
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=1,
        limit_val_batches=0,
        val_check_interval=0,
        num_sanity_val_steps=0,
    )


@track_io
class OrdTokenizer:
    def __init__(self, vocab_size=30_000, num_reserved_tokens=128, special_token_names=['bos_id', 'eos_id', 'pad_id']):
        self.vocab_size = vocab_size
        self.num_reserved_tokens = num_reserved_tokens
        self.special_token_names = special_token_names
        assert len(self.special_token_names) < num_reserved_tokens

    def __getattr__(self, name):
        if name in self.__dict__.get('special_token_names', {}):
            return self.__dict__['special_token_names'].index(name)
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError

    def text_to_ids(self, text):
        token_ids = list(map(lambda x: self.num_reserved_tokens + ord(x), list(text)))
        assert max(token_ids) < self.vocab_size
        return token_ids


def logger() -> nl.NeMoLogger:
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        every_n_train_steps=10,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return nl.NeMoLogger(
        name="nemo2_peft",
        log_dir="/tmp/peft_logs",
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=None,
    )


def squad(mbs, gbs) -> pl.LightningDataModule:
    return llm.SquadDataModule(seq_length=2048, micro_batch_size=mbs, global_batch_size=gbs, num_workers=0)


def get_mistral_expected_ckpt():
    return {
        "module.embedding.word_embeddings.weight": ("Tensor", [15104, 4096]),
        "module.decoder.layers.0.self_attention.core_attention._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.self_attention.linear_proj.weight": ("Tensor", [4096, 2048]),
        "module.decoder.layers.0.self_attention.linear_proj._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight": ("Tensor", [32, 2048]),
        "module.decoder.layers.0.self_attention.linear_proj.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.0.self_attention.linear_proj.adapter.linear_out.weight": ("Tensor", [2048, 32]),
        "module.decoder.layers.0.self_attention.linear_proj.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight": ("Tensor", [4096]),
        "module.decoder.layers.0.self_attention.linear_qkv.weight": ("Tensor", [3072, 4096]),
        "module.decoder.layers.0.self_attention.linear_qkv._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight": ("Tensor", [16, 4096]),
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight": ("Tensor", [3072, 32]),
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight": ("Tensor", [4096]),
        "module.decoder.layers.0.mlp.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight": ("Tensor", [16, 4096]),
        "module.decoder.layers.0.mlp.linear_fc1.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight": ("Tensor", [14336, 32]),
        "module.decoder.layers.0.mlp.linear_fc1.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.0.mlp.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.linear_fc2.adapter.linear_in.weight": ("Tensor", [32, 7168]),
        "module.decoder.layers.0.mlp.linear_fc2.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight": ("Tensor", [2048, 32]),
        "module.decoder.layers.0.mlp.linear_fc2.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.1.self_attention.core_attention._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.self_attention.linear_proj.weight": ("Tensor", [4096, 2048]),
        "module.decoder.layers.1.self_attention.linear_proj._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.self_attention.linear_proj.adapter.linear_in.weight": ("Tensor", [32, 2048]),
        "module.decoder.layers.1.self_attention.linear_proj.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.1.self_attention.linear_proj.adapter.linear_out.weight": ("Tensor", [2048, 32]),
        "module.decoder.layers.1.self_attention.linear_proj.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight": ("Tensor", [4096]),
        "module.decoder.layers.1.self_attention.linear_qkv.weight": ("Tensor", [3072, 4096]),
        "module.decoder.layers.1.self_attention.linear_qkv._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.self_attention.linear_qkv.adapter.linear_in.weight": ("Tensor", [16, 4096]),
        "module.decoder.layers.1.self_attention.linear_qkv.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.1.self_attention.linear_qkv.adapter.linear_out.weight": ("Tensor", [3072, 32]),
        "module.decoder.layers.1.self_attention.linear_qkv.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.1.mlp.linear_fc1.layer_norm_weight": ("Tensor", [4096]),
        "module.decoder.layers.1.mlp.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.linear_fc1.adapter.linear_in.weight": ("Tensor", [16, 4096]),
        "module.decoder.layers.1.mlp.linear_fc1.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.1.mlp.linear_fc1.adapter.linear_out.weight": ("Tensor", [14336, 32]),
        "module.decoder.layers.1.mlp.linear_fc1.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.1.mlp.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.linear_fc2.adapter.linear_in.weight": ("Tensor", [32, 7168]),
        "module.decoder.layers.1.mlp.linear_fc2.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.1.mlp.linear_fc2.adapter.linear_out.weight": ("Tensor", [2048, 32]),
        "module.decoder.layers.1.mlp.linear_fc2.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.final_layernorm.weight": ("Tensor", [4096]),
        "module.decoder.final_layernorm._extra_state": ("Tensor", [0]),
        "module.output_layer.weight": ("Tensor", [15104, 4096]),
        "module.output_layer._extra_state": ("NoneType", []),
    }


def get_mixtral_expected_ckpt():
    return {
        "module.embedding.word_embeddings.weight": ("Tensor", [15104, 4096]),
        "module.decoder.layers.0.self_attention.core_attention._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.self_attention.linear_proj.weight": ("Tensor", [4096, 2048]),
        "module.decoder.layers.0.self_attention.linear_proj._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight": ("Tensor", [32, 2048]),
        "module.decoder.layers.0.self_attention.linear_proj.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.0.self_attention.linear_proj.adapter.linear_out.weight": ("Tensor", [2048, 32]),
        "module.decoder.layers.0.self_attention.linear_proj.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight": ("Tensor", [4096]),
        "module.decoder.layers.0.self_attention.linear_qkv.weight": ("Tensor", [3072, 4096]),
        "module.decoder.layers.0.self_attention.linear_qkv._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight": ("Tensor", [16, 4096]),
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight": ("Tensor", [3072, 32]),
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.0.pre_mlp_layernorm.weight": ("Tensor", [4096]),
        "module.decoder.layers.0.pre_mlp_layernorm._extra_state": ("Tensor", [0]),
        "module.decoder.layers.0.mlp.router.weight": ("Tensor", [8, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.1.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.3.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.4.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.5.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.6.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.0.mlp.experts.local_experts.7.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.self_attention.core_attention._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.self_attention.linear_proj.weight": ("Tensor", [4096, 2048]),
        "module.decoder.layers.1.self_attention.linear_proj._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.self_attention.linear_proj.adapter.linear_in.weight": ("Tensor", [32, 2048]),
        "module.decoder.layers.1.self_attention.linear_proj.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.1.self_attention.linear_proj.adapter.linear_out.weight": ("Tensor", [2048, 32]),
        "module.decoder.layers.1.self_attention.linear_proj.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight": ("Tensor", [4096]),
        "module.decoder.layers.1.self_attention.linear_qkv.weight": ("Tensor", [3072, 4096]),
        "module.decoder.layers.1.self_attention.linear_qkv._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.self_attention.linear_qkv.adapter.linear_in.weight": ("Tensor", [16, 4096]),
        "module.decoder.layers.1.self_attention.linear_qkv.adapter.linear_in._extra_state": ("NoneType", []),
        "module.decoder.layers.1.self_attention.linear_qkv.adapter.linear_out.weight": ("Tensor", [3072, 32]),
        "module.decoder.layers.1.self_attention.linear_qkv.adapter.linear_out._extra_state": ("NoneType", []),
        "module.decoder.layers.1.pre_mlp_layernorm.weight": ("Tensor", [4096]),
        "module.decoder.layers.1.pre_mlp_layernorm._extra_state": ("Tensor", [0]),
        "module.decoder.layers.1.mlp.router.weight": ("Tensor", [8, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.1.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.2.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.3.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.4.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.6.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc1.weight": ("Tensor", [14336, 4096]),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc1._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc1.adapter.linear_in.weight": (
            "Tensor",
            [16, 4096],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc1.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc1.adapter.linear_out.weight": (
            "Tensor",
            [14336, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc1.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc2.weight": ("Tensor", [4096, 7168]),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc2._extra_state": ("Tensor", [4]),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc2.adapter.linear_in.weight": (
            "Tensor",
            [32, 7168],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc2.adapter.linear_in._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc2.adapter.linear_out.weight": (
            "Tensor",
            [2048, 32],
        ),
        "module.decoder.layers.1.mlp.experts.local_experts.7.linear_fc2.adapter.linear_out._extra_state": (
            "NoneType",
            [],
        ),
        "module.decoder.final_layernorm.weight": ("Tensor", [4096]),
        "module.decoder.final_layernorm._extra_state": ("Tensor", [0]),
        "module.output_layer.weight": ("Tensor", [15104, 4096]),
        "module.output_layer._extra_state": ("NoneType", []),
    }


def mixtral_8x7b() -> pl.LightningModule:
    tokenizer = OrdTokenizer()
    model = llm.MixtralModel(llm.MixtralConfig8x7B(num_layers=2), tokenizer=tokenizer)
    return model, get_mixtral_expected_ckpt()


def mistral_7b() -> pl.LightningModule:
    tokenizer = OrdTokenizer()
    model = llm.MistralModel(llm.MistralConfig7B(num_layers=2), tokenizer=tokenizer)
    return model, get_mistral_expected_ckpt()


def make_lora(use_exclude):
    if use_exclude:
        return llm.peft.LoRA(exclude_modules=['output_layer'], target_modules=[])
    else:
        return llm.peft.LoRA()


if __name__ == '__main__':
    args = get_args()
    if args.model == 'mistral':
        model, ref_ckpt = mistral_7b()
    else:
        model, ref_ckpt = mixtral_8x7b()

    lora = make_lora(args.use_exclude)

    llm.finetune(
        model=model,
        data=squad(args.mbs, args.gbs),
        trainer=trainer(args.tp, args.tp, args.ep, args.tp > 1, args.max_steps),
        peft=lora,
        log=logger(),
        optim=nl.MegatronOptimizerModule(
            config=OptimizerConfig(
                optimizer="adam",
                lr=0.0001,
                adam_beta2=0.98,
                use_distributed_optimizer=args.dist_opt,
                clip_grad=1.0,
                bf16=True,
            ),
        ),
    )
    if args.tp == 2:
        for key, val in model.state_dict().items():
            if isinstance(val, torch.Tensor):
                shape = list(val.size())
            else:
                shape = []
            if not '_extra_state' in key:
                assert ref_ckpt[key] == (f"{type(val).__name__}", shape)
