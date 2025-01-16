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
import os
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallelConfig as McoreDDPConfig
from megatron.core.transformer.enums import AttnBackend

from nemo.collections.llm import MixtralConfig8x3B, MixtralModel, PreTrainingDataModule
from nemo.collections.llm.api import train
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import MegatronStrategy, NeMoLogger, Trainer
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule as MegatronOptim
from nemo.lightning.pytorch.optim.megatron import OptimizerConfig


def tokenizer(vocab_path, merges_path):
    return get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=vocab_path,
        merges_file=merges_path,
    )


def load_dcp(ckpt_dir, torch_tensor=True):
    from pathlib import Path

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    fs_reader = FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == 'TensorStorageMetadata'
    }

    dcp.load(
        state_dict,
        storage_reader=fs_reader,
    )
    return state_dict


def main(args):
    strategy = MegatronStrategy(
        expert_model_parallel_size=args.devices,
        tensor_model_parallel_size=1,
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

    data = PreTrainingDataModule(
        args.data_path,
        seq_length=512,
        global_batch_size=2,
        micro_batch_size=1,
        num_workers=1,
        split='99,1,0',
        tokenizer=tokenizer(args.vocab_path, args.merges_path),
    )

    mixtral_config = MixtralConfig8x3B(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        num_query_groups=8,
        ffn_hidden_size=320,
        kv_channels=16,
        init_method_std=0.015,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        max_position_embeddings=512,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        attention_backend=AttnBackend.unfused,
    )
    mixtral_config.overlap_param_gather_with_optimizer_step = True

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
        log_num_zeros_in_grad=True,
        barrier_with_L1_time=True,
    )

    opt = MegatronOptim(config=optim_config)
    model = MixtralModel(mixtral_config, optim=opt, tokenizer=data.tokenizer)

    nemo_logger = NeMoLogger(
        name=args.experiment_name,
        use_datetime_version=False,
        explicit_log_dir=args.experiment_dir,
    )

    output_path = Path(args.experiment_dir)
    assert not output_path.exists(), f"Did not expect {output_path} to exist"

    train(
        model=model,
        resume=None,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )

    # Confirm checkpoint directory structure
    output_path = Path(args.experiment_dir) / "checkpoints/--None=0.0000-epoch=0-consumed_samples=8.0/weights"
    assert output_path.exists(), f"Expected {output_path} to exist"
    assert output_path.is_dir(), f"Expected {output_path} to be a directory"
    output_files = ['__0_0.distcp', '__0_1.distcp', 'common.pt', 'metadata.json', '.metadata']
    for file in output_files:
        path = output_path / file
        assert path.exists(), f"Expected {file} to exist"
        assert path.is_file(), f"Expected {file} to be a file"
        assert os.access(path, os.R_OK), f"Expected {file} to be readable"
        assert path.stat().st_size, f"Expected {file} to be non-empty"

    for file in os.listdir(output_path):
        assert file in output_files, f"Got unexpected {file} in checkpoint directory"

    # Finally confirm checkpoint contents
    expected_ckpt = {
        "module.embedding.word_embeddings.weight": (torch.Size([50304, 128]), torch.bfloat16, "cpu"),
        "module.decoder.layers.self_attention.linear_proj.weight": (torch.Size([2, 128, 128]), torch.bfloat16, "cpu"),
        "module.decoder.layers.self_attention.linear_qkv.layer_norm_weight": (
            torch.Size([2, 128]),
            torch.bfloat16,
            "cpu",
        ),
        "module.decoder.layers.self_attention.linear_qkv.weight": (torch.Size([2, 384, 128]), torch.bfloat16, "cpu"),
        "module.decoder.layers.pre_mlp_layernorm.weight": (torch.Size([2, 128]), torch.bfloat16, "cpu"),
        "module.decoder.layers.mlp.router.weight": (torch.Size([2, 8, 128]), torch.bfloat16, "cpu"),
        "module.decoder.layers.mlp.experts.experts.linear_fc1.weight": (
            torch.Size([2, 8, 640, 128]),
            torch.bfloat16,
            "cpu",
        ),
        "module.decoder.layers.mlp.experts.experts.linear_fc2.weight": (
            torch.Size([2, 8, 128, 320]),
            torch.bfloat16,
            "cpu",
        ),
        "module.decoder.final_layernorm.weight": (torch.Size([128]), torch.bfloat16, "cpu"),
        "module.output_layer.weight": (torch.Size([50304, 128]), torch.bfloat16, "cpu"),
        "optimizer.state.fp32_param.module.output_layer.weight": (torch.Size([1, 1, 6438912]), torch.float32, "cpu"),
        "optimizer.state.exp_avg.module.output_layer.weight": (torch.Size([1, 1, 6438912]), torch.float32, "cpu"),
        "optimizer.state.exp_avg_sq.module.output_layer.weight": (torch.Size([1, 1, 6438912]), torch.float32, "cpu"),
        "optimizer.state.fp32_param.module.decoder.final_layernorm.weight": (torch.Size([128]), torch.float32, "cpu"),
        "optimizer.state.exp_avg.module.decoder.final_layernorm.weight": (torch.Size([128]), torch.float32, "cpu"),
        "optimizer.state.exp_avg_sq.module.decoder.final_layernorm.weight": (torch.Size([128]), torch.float32, "cpu"),
        "optimizer.state.fp32_param.module.decoder.layers.mlp.experts.experts.linear_fc2.weight": (
            torch.Size([2, 8, 1, 1, 40960]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg.module.decoder.layers.mlp.experts.experts.linear_fc2.weight": (
            torch.Size([2, 8, 1, 1, 40960]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg_sq.module.decoder.layers.mlp.experts.experts.linear_fc2.weight": (
            torch.Size([2, 8, 1, 1, 40960]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.fp32_param.module.decoder.layers.mlp.experts.experts.linear_fc1.weight": (
            torch.Size([2, 8, 2, 1, 40960]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg.module.decoder.layers.mlp.experts.experts.linear_fc1.weight": (
            torch.Size([2, 8, 2, 1, 40960]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg_sq.module.decoder.layers.mlp.experts.experts.linear_fc1.weight": (
            torch.Size([2, 8, 2, 1, 40960]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.fp32_param.module.decoder.layers.mlp.router.weight": (
            torch.Size([2, 1, 1, 1024]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg.module.decoder.layers.mlp.router.weight": (
            torch.Size([2, 1, 1, 1024]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg_sq.module.decoder.layers.mlp.router.weight": (
            torch.Size([2, 1, 1, 1024]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.fp32_param.module.decoder.layers.pre_mlp_layernorm.weight": (
            torch.Size([2, 1, 128]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg.module.decoder.layers.pre_mlp_layernorm.weight": (
            torch.Size([2, 1, 128]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg_sq.module.decoder.layers.pre_mlp_layernorm.weight": (
            torch.Size([2, 1, 128]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.fp32_param.module.decoder.layers.self_attention.linear_qkv.weight": (
            torch.Size([2, 1, 1, 49152]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg.module.decoder.layers.self_attention.linear_qkv.weight": (
            torch.Size([2, 1, 1, 49152]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg_sq.module.decoder.layers.self_attention.linear_qkv.weight": (
            torch.Size([2, 1, 1, 49152]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.fp32_param.module.decoder.layers.self_attention.linear_qkv.layer_norm_weight": (
            torch.Size([2, 1, 128]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg.module.decoder.layers.self_attention.linear_qkv.layer_norm_weight": (
            torch.Size([2, 1, 128]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg_sq.module.decoder.layers.self_attention.linear_qkv.layer_norm_weight": (
            torch.Size([2, 1, 128]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.fp32_param.module.decoder.layers.self_attention.linear_proj.weight": (
            torch.Size([2, 1, 1, 16384]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg.module.decoder.layers.self_attention.linear_proj.weight": (
            torch.Size([2, 1, 1, 16384]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg_sq.module.decoder.layers.self_attention.linear_proj.weight": (
            torch.Size([2, 1, 1, 16384]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.fp32_param.module.embedding.word_embeddings.weight": (
            torch.Size([1, 1, 6438912]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg.module.embedding.word_embeddings.weight": (
            torch.Size([1, 1, 6438912]),
            torch.float32,
            "cpu",
        ),
        "optimizer.state.exp_avg_sq.module.embedding.word_embeddings.weight": (
            torch.Size([1, 1, 6438912]),
            torch.float32,
            "cpu",
        ),
    }
    ckpt = load_dcp(output_path)
    ckpt_keys = set(ckpt.keys())
    expected_keys = set(expected_ckpt.keys())
    assert len(ckpt) == len(expected_ckpt), (
        "Checkpoint length mismatch ",
        len(ckpt),
        len(expected_ckpt),
        ckpt_keys - expected_keys,
    )
    for key, (shape, dtype, device) in expected_ckpt.items():
        assert key in ckpt, f"Expected {key} to be in ckpt"
        assert isinstance(ckpt[key], torch.Tensor), f"Expected {key} to be a tensor"
        assert ckpt[key].shape == shape, f"Expected {key} shapes to match {ckpt[key].shape} & {shape}"
        assert ckpt[key].dtype == dtype, f"Expected {key} dtype to match {ckpt[key].dtype} & {dtype}"
        assert str(ckpt[key].device) == device, f"Expected {key} device to match {ckpt[key].device} & {device}"


def parse_args():
    parser = argparse.ArgumentParser(description='Train a small Mixtral model using NeMo 2.0')
    parser.add_argument('--devices', type=int, default=1, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, default=4, help="Number of steps to train for")
    parser.add_argument(
        '--experiment-dir', type=str, default='/tmp/exp_dir', help="directory to write results and checkpoints to"
    )
    parser.add_argument('--experiment-name', type=str, default='mini_mixtral_test', help="name of experiment")
    parser.add_argument('--data-path', type=str, help="Path to data file")
    parser.add_argument('--vocab-path', type=str, default=None, help="Path to vocab file")
    parser.add_argument('--merges-path', type=str, default=None, help="Path to merges file")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
