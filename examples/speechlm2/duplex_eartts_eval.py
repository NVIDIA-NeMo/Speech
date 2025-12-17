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

"""
Evaluation script for Duplex EARTTS models.

This script computes standard speech evaluation metrics for a given Duplex
EARTTS checkpoint, including Word Error Rate (WER), Character Error Rate (CER),
speaker encoder cosine similarity (SECS), and ASR BLEU score.

The configuration file must define a valid ``validation_ds`` based on a Lhotse
dataset using one of the following dataset formats:
- Duplex S2S standard format
- ``s2s_duplex_overlap_as_s2s_duplex``
- ``lhotse_magpietts_data_as_continuation``

During evaluation, the script saves generated audio samples to
``exp_manager.explicit_log_dir`` as specified in the configuration. For each
utterance, the following audio files may be produced:

- Autoregressive inference output (``*.wav``)
- Teacher-forced output (``*_tf.wav``)
- Ground-truth reference audio (``*_gt.wav``)

Args:
    config-path (str): Path to the directory containing the YAML configuration file.
    config-name (str): Name of the YAML configuration file.
    checkpoint_path (str): Path to the Duplex EARTTS checkpoint file.

Usage:
    python duplex_eartts_eval.py \
        --config-path=conf/ \
        --config-name=duplex_eartts.yaml \
        ++checkpoint_path=duplex_eartts_results/duplex_eartts/model.ckpt
"""

import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexEARTTSDataset

from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="duplex_eartts")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        if cfg.get("checkpoint_path", None):
            model = DuplexEARTTS.load_from_checkpoint(
                cfg.checkpoint_path,
                cfg=OmegaConf.to_container(cfg, resolve=True),
            )
        else:
            raise ValueError("For evaluation, you must provide `cfg.checkpoint_path`.")

    dataset = DuplexEARTTSDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        add_text_bos_and_eos_in_each_turn=cfg.data.get("add_text_bos_and_eos_in_each_turn", True),
        add_audio_prompt=cfg.data.get("add_audio_prompt", True),
        audio_prompt_duration=cfg.data.get("audio_prompt_duration", 3),
        num_delay_speech_tokens=cfg.model.get("num_delay_speech_tokens", 2),
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.validate(model, datamodule)


if __name__ == "__main__":
    inference()
