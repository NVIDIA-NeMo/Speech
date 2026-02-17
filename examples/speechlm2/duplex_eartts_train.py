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
import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexEARTTSDataset
from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from nemo.collections.speechlm2.parts.pretrained import load_checkpoint, set_model_dict_for_partial_init
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="duplex_eartts")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = DuplexEARTTS(OmegaConf.to_container(cfg, resolve=True))

        # load pretrained tts checkpoint if available
        if model.cfg.get("pretrained_tts_model", None):
            checkpoint_state = load_checkpoint(model.cfg.pretrained_tts_model)
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.tts_model.state_dict())
            model.tts_model.load_state_dict(checkpoint_state, strict=True)

        # load pretrained checkpoint and rescale the weights if needed
        if model.cfg.get("pretrained_model", None):
            model.restore_from_pretrained_checkpoint(model.cfg.pretrained_model)

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

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
