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

from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.models.vision_language_foundation.clip import MegatronCLIPModel
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


@hydra_runner(config_path="conf", config_name="megatron_clip_infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # These configs are required to be off during inference.
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.vision.precision = cfg.trainer.precision
        model_cfg.text.precision = cfg.trainer.precision
        if cfg.trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    trainer, model = setup_trainer_and_model_for_inference(
        model_provider=MegatronCLIPModel, cfg=cfg, model_cfg_modifier=model_cfg_modifier,
    )

    if model.cfg.get("megatron_amp_O2", False):
        vision_encoder = model.model.module.vision_encoder
        text_encoder = model.model.module.text_encoder
    else:
        vision_encoder = model.model.vision_encoder
        text_encoder = model.model.text_encoder

    val_image_transform, text_transform = get_preprocess_fns(model.cfg, model.tokenizer, is_train=False,)

    autocast_dtype = torch_dtype_from_precision(trainer.precision)

    image = Image.open(cfg.image_path).convert('RGB')
    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
    ):
        image = val_image_transform(image).unsqueeze(0).cuda()
        texts = text_transform(cfg.texts).cuda()
        image_features = vision_encoder(image)
        text_features = text_encoder(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    if is_global_rank_zero:
        print(f"Given image's CLIP text probability: ", list(zip(cfg.texts, text_probs[0].cpu().numpy())))


if __name__ == '__main__':
    main()
