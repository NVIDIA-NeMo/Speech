# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf.omegaconf import OmegaConf
from PIL import Image

from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform
from nemo.collections.multimodal.models.vision_language_foundation.megatron_nsfw_clip_models import (
    MegatronContentFilteringModel,
)
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


def _get_autocast_dtype(precision: str):
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    if precision in [32, "32", "32-true"]:
        return torch.float
    if precision in [16, "16", "16-mixed"]:
        return torch.half
    raise ValueError('precision must be in ["32-true", "16-mixed", "bf16-mixed"]')


@hydra_runner(config_path="conf", config_name="megatron_nsfw_infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # These configs are required to be off during inference.
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.vision.precision = cfg.trainer.precision
        if cfg.trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    trainer, model = setup_trainer_and_model_for_inference(
        model_provider=MegatronContentFilteringModel, cfg=cfg, model_cfg_modifier=model_cfg_modifier,
    )
    image_transform_fn = image_transform(
        (model.cfg.vision.img_h, model.cfg.vision.img_w),
        is_train=False,
        mean=model.cfg.vision.image_mean,
        std=model.cfg.vision.image_std,
        resize_longest_max=True,
    )

    autocast_dtype = _get_autocast_dtype(trainer.precision)
    image = Image.open(cfg.image_path).convert('RGB')
    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
    ):
        image = image_transform_fn(image).unsqueeze(0).cuda()
        probability = model(image).sigmoid()

    if is_global_rank_zero:
        print("Given image's NSFW probability: ", probability.cpu().item())


if __name__ == '__main__':
    main()
