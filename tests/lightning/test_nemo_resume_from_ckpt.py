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
import os
from typing import List, Optional

import pytest


def set_env():
    os.environ['NVTE_APPLY_QK_LAYER_SCALING'] = '0'


import sys
from pathlib import Path

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import AttnBackend

import nemo.lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

DATA_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document"
VOCAB_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json"
MERGES_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt"


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
        # no_dist=True,
    )
    return state_dict


def compare_ckpts(a, b, path: Optional[List[str]] = None):
    path = path if path is not None else []
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(b.keys())
        for key in a.keys():
            compare_ckpts(a[key], b[key], path + [key])
    elif isinstance(a, list):
        assert isinstance(b, list)
        assert len(a) == len(b)
        for i, (aa, bb) in enumerate(zip(a, b)):
            compare_ckpts(aa, bb, path + [f'[{i}]'])
    elif isinstance(a, torch.Tensor):
        skey = '.'.join(path)
        assert a.dtype == b.dtype, f"mismatch\t{skey}: different dtypes {a.dtype} {b.dtype}"
        assert a.shape == b.shape, f"mismatch\t{skey}: different shape {a.shape} {b.shape}"
        assert torch.all(a == b), f"mismatch\t{skey}: different values\n{a}\n{b}"
        print(f'match\t{skey}', file=sys.stderr)
    else:
        raise ValueError("Unexpected value type " + str(type(a)))


def setup_data(log_dir, n_steps, data_path, gbs=2, mbs=1):
    seq_length = 2048
    tokenizer = get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=VOCAB_PATH,
        merges_file=MERGES_PATH,
    )

    data = PreTrainingDataModule(
        paths=data_path,
        seq_length=2048,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        seed=1234,
        tokenizer=tokenizer,
        split='9999,1,1',
    )
    return data


def setup_model_optim(log_dir, n_steps, tokenizer, gbs=2, mbs=1):
    seq_length = 2048
    gpt_config = llm.GPTConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=1,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        normalization='RMSNorm',
        masked_softmax_fusion=False,
        attention_backend=AttnBackend.local,
    )

    model = llm.GPTModel(gpt_config, tokenizer=tokenizer)

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-2,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        clip_grad=1.0,
        log_num_zeros_in_grad=False,
        timers=None,
        bf16=True,
        use_distributed_optimizer=False,
    )
    optim = MegatronOptimizerModule(config=opt_config)

    return gpt_config, model, optim


def setup_trainer_and_logger(log_dir):

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        ckpt_include_optimizer=True,
        ckpt_parallel_load=True,
        ckpt_parallel_save_optim=False,
        ckpt_async_save=False,
        save_ckpt_format='torch_dist',
        progress_interval=1,
    )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=10,
        every_n_train_steps=10,
        always_save_context=True,
        save_context_on_train_end=True,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
        filename=f'{{step}}-{{epoch}}',
    )

    callbacks = [checkpoint_callback, TimingCallback()]

    trainer = nl.Trainer(
        devices=1,
        max_steps=40,
        accelerator="gpu",
        strategy=strategy,
        callbacks=callbacks,
        log_every_n_steps=1,
        val_check_interval=20,
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    nemo_logger = nl.NeMoLogger(
        log_dir=log_dir,
        version='v1',
        use_datetime_version=True,
        update_logger_directory=True,
        wandb=None,
        ckpt=checkpoint_callback,
    )

    return trainer, nemo_logger


def replace_first(x, old, new):
    assert x.startswith(old)
    return x.replace(old, new, 1)


def extract_model_keys(ckpt_keys):
    # should be a list or a set
    assert not isinstance(ckpt_keys, dict)
    return list(filter(lambda x: x.startswith('module.'), ckpt_keys))


def prepend_exp_avg(model_keys):
    return list(map(lambda x: replace_first(x, 'module.', 'optimizer.state.exp_avg.module.'), model_keys))


def prepend_exp_avg_sq(model_keys):
    return list(map(lambda x: replace_first(x, 'module.', 'optimizer.state.exp_avg_sq.module.'), model_keys))


def prepend_exp_avg_sq(model_keys):
    return list(map(lambda x: replace_first(x, 'module.', 'optimizer.state.fp32_param.module.'), model_keys))


def has_all_keys(ckpt_keys, keys):
    return all(map(lambda x: x in ckpt_keys, keys))


def teardown():
    import shutil

    for steps in [40, 10]:
        # if a directory does not exist, should not stop from removing another.
        try:
            shutil.rmtree(f'/tmp/mcore_logs_{steps}steps/')
        except:
            continue


class TestCkptStateRestoration:
    @pytest.mark.run_only_on('GPU')
    def test_resume_optim_state(self, tmp_path):
        def train(n_steps, resume):
            log_dir = f'/tmp/mcore_logs_{n_steps}steps'
            os.makedirs(log_dir, exist_ok=True)
            data_path = [DATA_PATH]
            data = setup_data(log_dir, n_steps, data_path, gbs=2, mbs=1)
            # Other tests might have different configs, so need to configure explicitly.
            from tests.lightning.mcore_microbatch_utils import reconfigure_num_microbatches_calculator_manager

            with reconfigure_num_microbatches_calculator_manager(
                0,
                None,
                2,  # gbs
                1,  # mbs
                data_parallel_size=1,
            ):
                gpt_config, model, optim = setup_model_optim(log_dir, n_steps, data.tokenizer)
                trainer, nemo_logger = setup_trainer_and_logger(log_dir)
                llm.train(
                    model=model,
                    data=data,
                    trainer=trainer,
                    log=nemo_logger,
                    resume=resume,
                    tokenizer='data',
                    optim=optim,
                )
                trainer._teardown()

        set_env()
        assert os.environ['NVTE_APPLY_QK_LAYER_SCALING'] == '0'

        # Train for 40 steps
        train(
            40,
            nl.AutoResume(
                resume_if_exists=True,
                resume_ignore_no_checkpoint=True,
            ),
        )

        # Train for 10 steps, resume from the 30th step of previous run.
        resume_path = '/tmp/mcore_logs_40steps/default/v1/checkpoints/step=29-epoch=0'
        assert Path(resume_path).exists()
        train(
            10,
            nl.AutoResume(
                resume_if_exists=True,
                resume_ignore_no_checkpoint=False,
                resume_from_path=resume_path,
            ),
        )

        # Finally check everything matches.
        paths = [
            '/tmp/mcore_logs_40steps/default/v1/checkpoints/step=39-epoch=0/weights',
            '/tmp/mcore_logs_10steps/default/v1/checkpoints/step=39-epoch=0/weights',
        ]
        assert all(map(lambda x: Path(x).exists(), paths))
        ckpts = list(map(load_dcp, paths))

        # Verify ckpt structure
        model_keys = extract_model_keys(ckpts[0].keys())
        assert len(model_keys) > 0
        assert set(model_keys) == set(extract_model_keys(ckpts[1].keys()))

        for ckpt_keys in [ckpts[0].keys(), ckpts[1].keys()]:
            assert has_all_keys(ckpt_keys, prepend_exp_avg(model_keys))
            assert has_all_keys(ckpt_keys, prepend_exp_avg_sq(model_keys))
            assert has_all_keys(ckpt_keys, prepend_exp_avg_sq(model_keys))

        # Verify ckpt contents
        compare_ckpts(ckpts[0], ckpts[1])
        teardown()
