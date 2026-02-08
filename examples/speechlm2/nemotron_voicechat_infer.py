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

from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset

from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model_config = OmegaConf.to_container(cfg, resolve=True)
        model = NemotronVoiceChat(model_config)

        # load pretrained checkpoint and rescale the weights if needed
        if model.tts_model.cfg.get("pretrained_model", None):
            model.tts_model.restore_from_pretrained_checkpoint(model.tts_model.cfg.pretrained_model)

    dataset = DuplexS2SDataset(
        tokenizer=model.stt_model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles
    )
    datamodule = DataModule(cfg.data, tokenizer=model.stt_model.tokenizer, dataset=dataset)
    # export file to huggingface
    hf_export_dir = model_config.get("hf_export_dir", None)
    if hf_export_dir:
        model.save_pretrained(hf_export_dir, config=model_config)
        print("Hugging face compatible checkpoint saved at:", hf_export_dir)

    trainer.validate(model, datamodule)


if __name__ == "__main__":
    inference()
