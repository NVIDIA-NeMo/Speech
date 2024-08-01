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
        # model_cfg.unet_config.from_pretrained = "/opt/nemo-aligner/checkpoints/sdxl/unet_nemo.ckpt"
        # model_cfg.unet_config.from_NeMo = True
        # model_cfg.first_stage_config.from_pretrained = "/opt/nemo-aligner/checkpoints/sdxl/vae_nemo.ckpt"
        # model_cfg.first_stage_config.from_NeMo = True
        model_cfg.first_stage_config._target_ = 'nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder.AutoencoderKLInferenceWrapper'
        # model_cfg.fsdp = True

    torch.backends.cuda.matmul.allow_tf32 = True
    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronDiffusionEngine, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )

    ### Manually configure sharded model
    # model = megatron_diffusion_model
    # model = trainer.strategy._setup_model(model)
    # model = model.cuda(torch.cuda.current_device())
    # get the diffusion part only
    model = megatron_diffusion_model.model
    model.cuda().eval()

    with torch.no_grad():
        base = SamplingPipeline(model, use_fp16=cfg.use_fp16, is_legacy=cfg.model.is_legacy)
        use_refiner = cfg.get('use_refiner', False)
        num_samples_per_batch = cfg.infer.get('num_samples_per_batch', cfg.infer.num_samples)
        num_batches = cfg.infer.num_samples // num_samples_per_batch

        for i, prompt in enumerate(cfg.infer.prompt):
            for batchid in range(num_batches):
                samples = base.text_to_image(
                    params=cfg.sampling.base,
                    prompt=[prompt],
                    negative_prompt=cfg.infer.negative_prompt,
                    samples=num_samples_per_batch,
                    return_latents=True if use_refiner else False,
                    seed=int(cfg.infer.seed + i * 100 + batchid * 200),
                )
                # samples=cfg.infer.num_samples,
                perform_save_locally(cfg.out_path, samples)


if __name__ == "__main__":
    main()
