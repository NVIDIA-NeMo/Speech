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
import torch.nn.functional as F
from omegaconf.omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from nemo.collections.multimodal.data.clip.clip_dataset import build_imagenet_validation_dataloader
from nemo.collections.multimodal.models.vision_language_foundation.clip import MegatronCLIPModel
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


@hydra_runner(config_path="conf", config_name="megatron_clip_imagenet_zeroshot")
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

    autocast_dtype = torch_dtype_from_precision(trainer.precision)

    with open_dict(cfg):
        cfg.model["vision"] = model.cfg.vision
        cfg.model["text"] = model.cfg.text

    imagenet_val = build_imagenet_validation_dataloader(cfg.model, model.tokenizer)
    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
    ):
        # build imagenet classification classifier
        classifier = []
        for texts in imagenet_val["texts"]:
            texts = texts.cuda(non_blocking=True)
            class_embeddings = text_encoder(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            classifier.append(class_embedding)
        classifier = torch.stack(classifier, dim=1)

        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(imagenet_val["images"], desc="Imagenet Zero-shot Evaluation", leave=False):
            if images is None or target is None:
                continue

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # predict
            image_features = vision_encoder(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

        logging.info('Finished zero-shot imagenet.')
        top1 = top1 / n
        top5 = top5 / n

    imagenet_metric = torch.zeros(2).cuda()
    imagenet_metric[0], imagenet_metric[1] = top1, top5
    imagenet_metric = average_losses_across_data_parallel_group(imagenet_metric)

    if is_global_rank_zero:
        logging.info(f"Zero-shot CLIP accuracy Top-1: {imagenet_metric[0]:.4f}; Top-5: {imagenet_metric[1]:.4f}")


if __name__ == '__main__':
    main()
