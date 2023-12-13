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

import os

import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.collections.vision.data.megatron.image_folder import ImageFolder
from nemo.collections.vision.data.megatron.vit_dataset import ClassificationTransform
from nemo.collections.vision.models.megatron_vit_classification_models import MegatronVitClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


@hydra_runner(config_path="conf", config_name="megatron_vit_classification_evaluate")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True, find_unused_parameters=False,  # we don't use DDP for async grad allreduce
    )
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

    model_cfg = MegatronVitClassificationModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        save_restore_connector=save_restore_connector,
        return_config=True,
    )

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == model_cfg.tensor_model_parallel_size * model_cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    # These configs are required to be off during inference.
    with open_dict(model_cfg):
        model_cfg.precision = trainer.precision
        if trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    model = MegatronVitClassificationModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=model_cfg,
        save_restore_connector=save_restore_connector,
        strict=True,
    )

    model.eval()

    val_transform = ClassificationTransform(model.cfg, (model.cfg.img_h, model.cfg.img_w), train=False)
    val_data = ImageFolder(root=cfg.model.data.imagenet_val, transform=val_transform,)

    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    test_loader = DataLoader(val_data, batch_size=cfg.model.micro_batch_size, num_workers=cfg.model.data.num_workers,)

    autocast_dtype = torch_dtype_from_precision(trainer.precision)

    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
    ):
        total = correct = 0.0
        for tokens, labels in tqdm(test_loader):
            logits = model(tokens.cuda())
            class_indices = torch.argmax(logits, -1)
            correct += (class_indices == labels.cuda()).float().sum()
            total += len(labels)

    if is_global_rank_zero:
        print(f"ViT Imagenet 1K Evaluation Accuracy: {correct / total:.4f}")


if __name__ == '__main__':
    main()
