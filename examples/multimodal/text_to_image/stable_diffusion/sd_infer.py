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
import torch

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.multimodal.parts.stable_diffusion.pipeline import pipeline
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner


@hydra_runner(config_path='conf', config_name='sd_infer')
def main(cfg):
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.use_flash_attention = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None
        model_cfg.first_stage_config._target_ = (
            'nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder.AutoencoderKL'
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronLatentDiffusion, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )
    model = megatron_diffusion_model.model
    model.cuda().eval()

    rng = torch.Generator().manual_seed(cfg.infer.seed)
    pipeline(model, cfg, rng=rng)


if __name__ == "__main__":
    main()
