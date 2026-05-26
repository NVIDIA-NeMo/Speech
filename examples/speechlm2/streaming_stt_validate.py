# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass
from typing import Optional
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, StreamingSTTDataset, StreamingSTTModel
from nemo.core.classes.common import Serialization
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@dataclass
class StreamingSTTEvalConfig:
    pretrained_name: str
    inputs: str
    batch_size: int = 2
    max_new_tokens: int = 64
    system_prompt: str = "Transcribe the audio into text."
    system_role: str = "system"
    simulate_streaming: bool = False
    output_manifest: Optional[str] = "streaming_stt_generations.jsonl"
    verbose: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_normalizer: Optional[str] = "english"  # "english", "basic", or "none"
    num_delay_frames: int = 1


@hydra_runner(config_name="StreamingSTTEvalConfig", schema=StreamingSTTEvalConfig)
def main(cfg: StreamingSTTEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    model = StreamingSTTModel.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)

    trainer = Trainer()
    # Inherit dataset-relevant fields from the trained model. Critically,
    # compact_template and write_token MUST match training — otherwise the
    # dataset tokenizes inputs in a different chat-template layout than the
    # model expects, and every val metric (val_acc, val_aux_*, val_loss) is
    # garbage even with the correct checkpoint loaded.
    dataset_cfg = OmegaConf.create(
        {
            "sample_rate": model.sampling_rate,
            "frame_length_in_secs": model.core_cfg.frame_length_in_secs,
            "chunk_size": model.core_cfg.chunk_size,
            "num_delay_frames": cfg.num_delay_frames,
            "audio_tag": model.core_cfg.audio_tag,
            "blank_token": model.core_cfg.blank_token,
            "system_role": cfg.system_role,
            "system_prompt": cfg.system_prompt,
            "compact_template": model.core_cfg.compact_template,
            "write_token": model.core_cfg.write_token,
        }
    )
    dataset = StreamingSTTDataset(cfg=dataset_cfg, tokenizer=model.tokenizer)

    data_cfg = {
        "manifest_filepath": cfg.inputs,
        "sample_rate": model.sampling_rate,
        "batch_size": cfg.batch_size,
        "num_workers": 4,
        "shuffle": False,
    }
    data_cfg = OmegaConf.create({"validation_ds": data_cfg})
    datamodule = DataModule(data_cfg, tokenizer=model.tokenizer, dataset=dataset)
    trainer.validate(model, datamodule)


if __name__ == "__main__":
    main()
