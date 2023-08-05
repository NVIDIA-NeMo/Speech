# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.nlp.data.machine_translation.preproc_mt_data import MTDataPreproc
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.config_utils import update_model_config
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager


"""
Usage:
 1. If you need to start docker and install NeMo, otherwise skip this step:
 
    a. ```docker run --gpus all -it --rm -v /home/okuchaiev/repos/NeMo/:/NeMo -p 6006:6006  -v /mnt:/mnt --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:20.11-py3```
    b. ```cd /NeMo```
    c. ```./reinstall.sh```
 
 2. Train a new tokenizer (or use pre-trained one):
    ```yttm bpe --data /mnt/D1/Data/NMT/wmt16_de_en/train.clean.en-de.shuffled.common --model tokenizer.BPE.8192.model --vocab_size 8192```

(To use WANDB, optionally, do login first)
``wandb login [YOUR WANDB login]``
    
 3. Start training:
 

 (This example for "base" model on 2 GPUs for 150000 steps with batch size of 12500 tokens per GPU)
 
 python enc_dec_nmt.py \
      --config-path=conf \
      --config-name=aayn_base \
      trainer.devices=[0,1] \
      ~trainer.max_epochs \
      +trainer.max_steps=150000 \
      model.beam_size=4 \
      model.max_generation_delta=5 \
      model.label_smoothing=0.1 \
      model.encoder_tokenizer.tokenizer_model=tokenizer.BPE.8192.model  \
      model.decoder_tokenizer.tokenizer_model=tokenizer.BPE.8192.model  \
      model.encoder.num_layers=6 \
      model.encoder.hidden_size=512 \
      model.encoder.inner_size=2048 \
      model.encoder.num_attention_heads=8 \
      model.encoder.ffn_dropout=0.1 \
      model.decoder.num_layers=6 \
      model.decoder.hidden_size=512 \
      model.decoder.inner_size=2048 \
      model.decoder.num_attention_heads=8 \
      model.decoder.ffn_dropout=0.1 \
      model.train_ds.src_file_name=/mnt/D1/Data/NMT/wmt16_de_en/train.clean.de.shuffled \
      model.train_ds.tgt_file_name=/mnt/D1/Data/NMT/wmt16_de_en/train.clean.en.shuffled \
      model.train_ds.tokens_in_batch=12500 \
      model.validation_ds.src_file_name=/mnt/D1/Data/NMT/wmt16_de_en/wmt14-en-de.ref \
      model.validation_ds.tgt_file_name=/mnt/D1/Data/NMT/wmt16_de_en/wmt14-en-de.src \
      model.validation_ds.tokens_in_batch=8192 \
      model.test_ds.src_file_name=/mnt/D1/Data/NMT/wmt16_de_en/wmt14-en-de.ref \
      model.test_ds.tgt_file_name=/mnt/D1/Data/NMT/wmt16_de_en/wmt14-en-de.src \
      model.optim.lr=0.001  \
      model.optim.sched.warmup_ratio=0.05 \
      +exp_manager.create_wandb_logger=True \
      +exp_manager.wandb_logger_kwargs.name=TEST-nmt-base \
      +exp_manager.wandb_logger_kwargs.project=nmt-de-en \
      +exp_manager.create_checkpoint_callback=True \
      +exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
      +exp_manager.exp_dir=nmt_base \
      +exp_manager.checkpoint_callback_params.mode=max 
"""


@dataclass
class MTEncDecConfig(NemoConfig):
    name: Optional[str] = 'MTEncDec'
    do_training: bool = True
    do_testing: bool = False
    model: MTEncDecModelConfig = MTEncDecModelConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MTEncDec', files_to_copy=[])


@hydra_runner(config_path="conf", config_name="aayn_base")
def main(cfg: MTEncDecConfig) -> None:
    # merge default config with user specified config
    default_cfg = MTEncDecConfig()
    cfg = update_model_config(default_cfg, cfg)
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')

    # training is managed by PyTorch Lightning
    trainer_cfg = OmegaConf.to_container(cfg.trainer)
    trainer_cfg.pop('strategy', None)
    trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_cfg)

    # tokenizers will be trained and and tarred training data will be created if needed
    # model config is then updated
    if cfg.model.preproc_out_dir is not None:
        MTDataPreproc(cfg=cfg.model, trainer=trainer)

    # experiment logs, checkpoints, and auto-resume are managed by exp_manager and PyTorch Lightning
    exp_manager(trainer, cfg.exp_manager)

    # everything needed to train translation models is encapsulated in the NeMo MTEncdDecModel
    mt_model = MTEncDecModel(cfg.model, trainer=trainer)

    logging.info("\n\n************** Model parameters and their sizes ***********")
    for name, param in mt_model.named_parameters():
        print(name, param.size())
    logging.info("***********************************************************\n\n")

    if cfg.do_training:
        trainer.fit(mt_model)
    # Reset for PTL 2.0 as test uses the same ckpt as train via previously set self._ckpt_path
    trainer.ckpt_path = None
    if cfg.do_testing:
        trainer.test(mt_model)


if __name__ == '__main__':
    main()
