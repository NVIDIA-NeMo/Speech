import os
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator

import nemo.lightning as nl
from nemo.collections import llm
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO, AsyncFinalizerCallback


def _get_strategy():
    strategy = nl.MegatronStrategy(
        enable_nemo_ckpt_io=False,
    )
    return strategy


def _get_last_checkpoint_dir(model: pl.LightningModule, suffix: str = '') -> Path:
    return f'epoch={model.trainer.current_epoch - 1}-step={model.trainer.max_steps - 1}{suffix}'


def get_model_and_data():
    micro_batch_size = 2
    global_batch_size = 2
    seq_length = 128
    data = llm.MockDataModule(
        seq_length=seq_length, micro_batch_size=micro_batch_size, global_batch_size=global_batch_size
    )

    config = llm.GPTConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        seq_length=seq_length,
        apply_query_key_layer_scaling=1,
    )
    reconfigure_num_microbatches_calculator(
        0,
        None,
        global_batch_size,
        micro_batch_size,
        data_parallel_size=1,
    )
    return llm.GPTModel(config, tokenizer=data.tokenizer), data


class TestDistCkptIO:

    @pytest.mark.run_only_on('GPU')
    def test_dist_ckpt_io_called_for_mcore_models(self, tmp_path):

        model, data = get_model_and_data()

        strategy = _get_strategy()

        trainer = nl.Trainer(
            devices=1,
            accelerator="gpu",
            strategy=strategy,
            enable_checkpointing=True,
            max_steps=2,
            default_root_dir=str(tmp_path),
            logger=False,
        )

        trainer.fit(model, data)

        assert isinstance(trainer.strategy.checkpoint_io, MegatronCheckpointIO)
        # Ckpt path doesn't contain the .ckpt suffix
        ckpts = os.listdir(Path(tmp_path / "checkpoints"))
        assert len(ckpts) == 1
        ckpt = ckpts[0]
        assert str(ckpt) == _get_last_checkpoint_dir(model)

    @pytest.mark.run_only_on('GPU')
    def test_async_save_produces_same_checkpoints_as_sync(self, tmp_path):

        model, data = get_model_and_data()

        sync_ckpt_dir = tmp_path / 'sync_checkpoints'
        async_ckpt_dir = tmp_path / 'async_checkpoints'

        sync_checkpoint_io = MegatronCheckpointIO('torch_dist')
        async_checkpoint_io = AsyncFinalizableCheckpointIO(MegatronCheckpointIO('torch_dist', async_save=True))

        # dummy_trainer just to initialize NCCL
        dummy_trainer = pl.Trainer(
            devices=1,
            logger=False,
            max_steps=2,
            strategy=_get_strategy(),
        )
        dummy_trainer.fit(model, data)
        strategy = _get_strategy()
        tmp_path = strategy.broadcast(tmp_path)

        ## reset the model and data and train with sync checkpointing
        model, data = get_model_and_data()
        sync_test_trainer = pl.Trainer(
            devices=1,
            enable_checkpointing=True,
            logger=False,
            max_steps=2,
            strategy=_get_strategy(),
            plugins=[sync_checkpoint_io],
            default_root_dir=str(sync_ckpt_dir),
        )
        sync_test_trainer.fit(model, data)

        ## reset the model and data and train with sync checkpointing
        model, data = get_model_and_data()
        async_test_trainer = pl.Trainer(
            devices=1,
            enable_checkpointing=True,
            logger=False,
            max_steps=2,
            strategy=_get_strategy(),
            plugins=[async_checkpoint_io],
            callbacks=AsyncFinalizerCallback(),
            default_root_dir=str(async_ckpt_dir),
        )
        async_test_trainer.fit(model, data)

        checkpoint = {'sharded_state_dict': model.sharded_state_dict()}

        sync_state_dict = sync_checkpoint_io.load_checkpoint(
            Path(f"{sync_ckpt_dir}/checkpoints/{_get_last_checkpoint_dir(model)}"), sharded_state_dict=checkpoint
        )

        async_state_dict = async_checkpoint_io.load_checkpoint(
            Path(f"{async_ckpt_dir}/checkpoints/{_get_last_checkpoint_dir(model)}"), sharded_state_dict=checkpoint
        )

        ## one of the keys is a _io.BytesIO object
        for k in sync_state_dict['sharded_state_dict'].keys():
            if isinstance(sync_state_dict['sharded_state_dict'][k], torch.Tensor):
                assert torch.all(sync_state_dict['sharded_state_dict'][k] == async_state_dict['sharded_state_dict'][k])

    def test_sharded_strategies(self):

        model_checkpoint = nl.ModelCheckpoint()

        strategy = nl.MegatronStrategy(
            enable_nemo_ckpt_io=False,
            save_ckpt_format='torch_dist',
            ckpt_parallel_save=True,
            ckpt_load_directly_on_device=False,
            ckpt_async_save=True,
        )
        trainer = nl.Trainer(
            callbacks=[model_checkpoint],
            strategy=strategy,
        )

        assert isinstance(strategy.checkpoint_io, AsyncFinalizableCheckpointIO)
        assert isinstance(strategy.checkpoint_io._checkpoint_io, MegatronCheckpointIO)

        base_checkpoint_io = strategy.checkpoint_io._checkpoint_io

        assert base_checkpoint_io.save_ckpt_format == 'torch_dist'
        assert base_checkpoint_io.parallel_save
        assert base_checkpoint_io.load_directly_on_device == False
