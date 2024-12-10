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

## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import vlm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.api import train
from nemo.lightning import AutoResume, NeMoLogger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, ParameterDebugger
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule


def get_args():
    parser = argparse.ArgumentParser(description='Train a small MLLAMA model using NeMo 2.0')
    parser.add_argument('--devices', type=int, default=1, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, default=5, help="Number of steps to train for")
    parser.add_argument(
        '--experiment-dir', type=str, default=None, help="directory to write results and checkpoints to"
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    gbs = 4
    mbs = 1
    # encoder (vision) seq length
    # ((img_res / patch_size) ** 2 + cls_token) * num_tiles, = ((560 / 14) ** 2 + 1) * 4 = 6404
    seq_length = 6404
    decoder_seq_length = 768  # decoder (llm) seq length
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    image_processor = processor.image_processor
    tokenizer = AutoTokenizer(model_id)

    data = vlm.MLlamaMockDataModule(
        seq_length=seq_length,
        decoder_seq_length=decoder_seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        num_workers=2,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    # Transformer configurations
    conf = vlm.MLlamaConfig11BInstruct()
    conf.language_model_config.num_layers = 2
    # conf.vision_model_config.num_layers = 8
    model = vlm.MLlamaModel(conf, tokenizer=tokenizer)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        encoder_pipeline_model_parallel_size=0,
        pipeline_dtype=torch.bfloat16,
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=5000,
        save_optim_on_train_end=True,
    )

    def create_verify_precision(precision: torch.dtype):
        def verify_precision(tensor: torch.Tensor) -> None:
            assert tensor.dtype == precision

        return verify_precision

    debugger = ParameterDebugger(
        param_fn=create_verify_precision(torch.bfloat16),
        grad_fn=create_verify_precision(torch.float32),
        log_on_hooks=["on_train_start", "on_train_end"],
    )
    callbacks = [checkpoint_callback, debugger]

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
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    nemo_logger = NeMoLogger(
        log_dir=args.experiment_dir,
    )

    resume = AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )

    train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume,
        tokenizer='data',
        optim=opt,
    )
