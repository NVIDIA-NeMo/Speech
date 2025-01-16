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
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallelConfig as McoreDDPConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import init_method_normal, scaled_init_method_normal

from nemo.collections.llm import MixtralConfig8x7B, MixtralModel, PreTrainingDataModule
from nemo.collections.llm.api import train
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import MegatronStrategy, NeMoLogger, Trainer
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule as MegatronOptim
from nemo.lightning.pytorch.optim.megatron import OptimizerConfig

VOCAB_PATH = '/mnt/4tb/gpt_tokenizer/vocab.json'
MERGES_PATH = '/mnt/4tb/gpt_tokenizer/merges.txt'
DATA_PATH = '/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document'


def tokenizer(vocab_path, merges_path):
    return get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=vocab_path,
        merges_file=merges_path,
    )


def main(args):
    strategy = MegatronStrategy(
        tensor_model_parallel_size=args.devices,
        sequence_parallel=False,
        context_parallel_size=1,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.float32,
        precision=torch.bfloat16,
        ddp=McoreDDPConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
            check_for_nan_in_grad=True,
            # mcore bucket_size is based on num of parameters, therefore not
            # using bucket_cap_mb to configure bucket_size here
            bucket_size=None,
        ),
    )

    trainer = Trainer(
        log_every_n_steps=1,
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        num_sanity_val_steps=0,
        logger=None,
        limit_val_batches=1,
    )

    init_method = scaled_init_method_normal(0.008, num_layers=2)
    mixtral_config = MixtralConfig8x7B(
        init_method_std=0.008,
        output_layer_init_method=init_method,
        init_method=init_method_normal(0.008),
        kv_channels=8,
        layernorm_zero_centered_gamma=True,
        max_position_embeddings=None,
        moe_aux_loss_coeff=0.0,
        moe_router_topk=1,
        moe_token_dispatcher_type='allgather',
        normalization="LayerNorm",
        num_attention_heads=4,
        num_layers=2,
        num_moe_experts=2,
        hidden_size=32,
        ffn_hidden_size=64,
        num_query_groups=4,
        persist_layer_norm=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype="torch.bfloat16",
        autocast_dtype="torch.bfloat16",
        use_cpu_initialization=None,
        cross_entropy_loss_fusion=False,
        deallocate_pipeline_outputs=True,
        bias_activation_fusion=True,
        async_tensor_model_parallel_allreduce=True,
        gradient_accumulation_fusion=False,
        bias_dropout_fusion=True,
        apply_rope_fusion=True,
        distribute_saved_activations=False,
        attention_backend=AttnBackend.unfused,
    )

    data = PreTrainingDataModule(
        DATA_PATH,
        seq_length=4096,
        global_batch_size=2,
        micro_batch_size=1,
        num_workers=1,
        split='99,1,0',
        tokenizer=tokenizer(args.vocab_path, args.merges_path),
    )

    optim_config = OptimizerConfig(
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        lr=0.01,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.9,
        clip_grad=0.0,
        use_distributed_optimizer=True,
        min_lr=0.0,
        # overlap_grad_reduce=False,
        # overlap_param_gather=False,
        log_num_zeros_in_grad=True,
        barrier_with_L1_time=True,
    )

    opt = MegatronOptim(config=optim_config)
    model = MixtralModel(mixtral_config, optim=opt, tokenizer=data.tokenizer)

    nemo_logger = NeMoLogger(
        name=args.exp_name,
        use_datetime_version=False,
        explicit_log_dir=args.exp_dir,
    )

    train(
        model=model,
        resume=None,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train a small Mixtral model using NeMo 2.0')
    parser.add_argument('--devices', type=int, default=1, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, default=10, help="Number of steps to train for")
    parser.add_argument(
        '--exp-dir', type=str, default='/tmp/exp_dir', help="directory to write results and checkpoints to"
    )
    parser.add_argument('--exp-name', type=str, default='mini_mixtral_test', help="name of experiment")
    parser.add_argument('--data-path', type=str, default=DATA_PATH, help="Path to data file")
    parser.add_argument('--vocab-path', type=str, default=VOCAB_PATH, help="Path to vocab file")
    parser.add_argument('--merges-path', type=str, default=MERGES_PATH, help="Path to merges file")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
