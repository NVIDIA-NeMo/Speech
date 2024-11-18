# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import tempfile
from typing import Any, Dict

import torch
from lightning.pytorch import Trainer

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner


def load_lora(lora_nemo):

    with tempfile.TemporaryDirectory() as tmpdir:
        NLPSaveRestoreConnector._unpack_nemo_file(lora_nemo, tmpdir)
        # assert os.path.isdir(lora_extracted_dir), "requires the untar'ed the lora .nemo file"

        ckpt_file = f"{tmpdir}/model_weights.ckpt"

        lora_state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
        return lora_state_dict


def merge(base_model_state_dict: Dict[str, Any], lora_state_dict: Dict[int, Any], lora_scale=1.0):

    for key in lora_state_dict.keys():
        if 'linear_out' in key:
            continue
        key_lora_in = key
        key_lora_out = key.replace('linear_in', 'linear_out')
        key_base_model = key.replace('.adapter_layer.parallel_linear_adapter.linear_in', '').replace('._orig_mod', '')

        wt_lora_in = lora_state_dict[key_lora_in]
        wt_lora_out = lora_state_dict[key_lora_out]
        wt_base_model = base_model_state_dict[key_base_model]

        wt_lora = wt_lora_out @ wt_lora_in
        base_model_state_dict[key_base_model] = wt_base_model + wt_lora * lora_scale
        print(f"merging weights for following key : {key_base_model}")
    return base_model_state_dict


@hydra_runner(config_path="conf", config_name="merge_lora_weights")
def main(cfg) -> None:
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.use_flash_attention = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None

    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronLatentDiffusion, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )
    model = megatron_diffusion_model.cpu()
    lora_weights = load_lora(cfg.lora_model_path)

    merged_weights = merge(model.state_dict(), lora_weights, lora_scale=cfg.lora_scale)

    model.load_state_dict(merged_weights)

    model.save_to(cfg.merged_model_path)
    print(f"saved merged model to {cfg.merged_model_path}")


if __name__ == '__main__':
    main()
