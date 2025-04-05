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


import random

import megatron.core.parallel_state as ps
import torch
import torch.distributed as dist
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, get_model, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

import nemo.lightning as nl
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.utils import logging

"""
This works on 1 device with no parallelism. With model parallelism, forward pass is not deterministic.
To run this test, you need to set the following environment variables:

export NVTE_FLASH_ATTN=1; export NVTE_FUSED_ATTN=0; export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc-per-node 1 /opt/NeMo/tests/collections/llm/gpt/model/test_nm5_nemo2_mlm_accuracy_match.py \
            --use-checkpoint-args \
            --load /lustre/fsw/portfolios/llmservice/users/kezhik/nemotron-5.1-hybrid/nemotron5/8b_hybrid/checkpoints/1t-hybrid-phase3/ \
            --bf16 \
            --tensor-model-parallel-size 1 \
            --attention-backend flash \
            --micro-batch-size 2 \
            --nemo-model-path /lustre/fsw/portfolios/coreai/users/ataghibakhsh/final_nm5/nm5_8b_base_8k_phase3 \
            --sequence-length 512
"""


def add_test_args(parser):
    group = parser.add_argument_group(title='more test args')
    group.add_argument("--nemo-model-path", type=str, required=True, help='Path to the NeMo2 checkpoint.')
    group.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help='Maximum sequence length for the model input.',
    )

    return parser


def megatron_model_provider(pre_process=True, post_process=True) -> MambaModel:
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        MambaModel: The returned model
    """
    args = get_args()

    print_rank_0('building Mamba model ...')
    config = core_transformer_config_from_args(get_args())

    assert args.use_legacy_models == False, "Mamba only supported in Mcore!"

    if args.spec is not None:
        mamba_stack_spec = import_module(args.spec)
    else:
        raise ValueError("You must provide a valid Mamba layer spec!")

    model = MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=False,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )

    return model


if __name__ == "__main__":

    initialize_megatron(
        extra_args_provider=add_test_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()
    args.exit_on_missing_checkpoint = True

    ##########
    data = [random.randint(0, 100000) for _ in range(args.sequence_length)]
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((args.micro_batch_size, 1)).cuda()
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((args.micro_batch_size, 1)).cuda()
    attention_mask = None
    ##########

    megatron_model = get_model(megatron_model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(megatron_model, None, None)

    assert len(megatron_model) == 1, "Above condition should have caught this"

    megatron_model = megatron_model[0]
    megatron_model.eval()
    megatron_out = megatron_model.forward(
        input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
            fp8=None,
            fp8_amax_history_len=1,
            fp8_amax_compute_algo="max",
        ),
    )

    nemo_model: nl.io.TrainerContext = nl.io.load_context(
        path=ckpt_to_context_subdir(args.nemo_model_path), subpath="model"
    )
    _setup_trainer_and_restore_model(path=args.nemo_model_path, trainer=trainer, model=nemo_model)
    nemo_model.eval()
    nemo_out = nemo_model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    assert (
        abs(megatron_out - nemo_out).sum().item() == 0
    ), "ERROR: nemo and megatron lm outputs are NOT bitwise similar! "
    logging.info("SUCCESS: The outputs of nemo and mcore models are bitwise similar!")
    ps.destroy_model_parallel()
    dist.destroy_process_group()
