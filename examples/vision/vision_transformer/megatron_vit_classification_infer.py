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

import glob
import os

import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.collections.vision.data.imagenet_classnames import imagenet_classnames
from nemo.collections.vision.data.megatron.vit_dataset import ClassificationTransform
from nemo.collections.vision.models.megatron_vit_classification_models import MegatronVitClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm".split()


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        # Use glob to find all image files in folder_path
        image_paths = []
        for ext in _IMG_EXTENSIONS + [x.upper() for x in _IMG_EXTENSIONS]:
            search_pattern = os.path.join(folder_path, f"*.{ext}")
            image_paths += glob.glob(search_pattern)
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


@hydra_runner(config_path="conf", config_name="megatron_vit_classification_infer")
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

    test_transform = ClassificationTransform(cfg.model, (model_cfg.img_h, model_cfg.img_w), train=False)
    test_data = ImageFolderDataset(folder_path=cfg.data_path, transform=test_transform,)
    test_loader = DataLoader(test_data, batch_size=8)

    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    autocast_dtype = torch_dtype_from_precision(trainer.precision)

    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
    ):
        class_names = []
        for tokens in test_loader:
            logits = model(tokens.cuda())
            class_indices = torch.argmax(logits, -1)
            class_names += [imagenet_classnames[x] for x in class_indices]

    if is_global_rank_zero:
        filenames = [os.path.basename(f) for f in test_data.image_paths]
        print(f"Predicted classes: ", list(zip(filenames, class_names)))


if __name__ == '__main__':
    main()
