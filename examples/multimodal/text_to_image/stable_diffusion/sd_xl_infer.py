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

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import MegatronDiffusionEngine
from nemo.collections.multimodal.parts.stable_diffusion.sdxl_helpers import perform_save_locally
from nemo.collections.multimodal.parts.stable_diffusion.sdxl_pipeline import SamplingPipeline
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner


@hydra_runner(config_path='conf', config_name='sd_xl_infer')
def main(cfg):
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None
        model_cfg.first_stage_config._target_ = 'nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder.AutoencoderKLInferenceWrapper'
        model_cfg.fsdp = False

    torch.backends.cuda.matmul.allow_tf32 = True
    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronDiffusionEngine, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )

    model = megatron_diffusion_model.model
    model.cuda().eval()

    base = SamplingPipeline(model, use_fp16=cfg.use_fp16, is_legacy=cfg.model.is_legacy)
    use_refiner = cfg.get('use_refiner', False)
    for i, prompt in enumerate(cfg.infer.prompt):
        samples = base.text_to_image(
            params=cfg.sampling.base,
            prompt=[prompt],
            negative_prompt=cfg.infer.negative_prompt,
            samples=cfg.infer.num_samples,
            return_latents=True if use_refiner else False,
            seed=int(cfg.infer.seed + i * 100),
        )

        perform_save_locally(cfg.out_path, samples)


if __name__ == "__main__":
    main()
